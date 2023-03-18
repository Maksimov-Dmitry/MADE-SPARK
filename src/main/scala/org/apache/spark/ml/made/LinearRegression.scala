package org.apache.spark.ml.made

import org.apache.spark.ml.param.shared.{HasFeaturesCol, HasLabelCol, HasPredictionCol}
import org.apache.spark.sql.types.StructType
import org.apache.spark.ml.linalg.{DenseVector, Vector, VectorUDT, Vectors}
import org.apache.spark.ml.util._
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.sql.{DataFrame, Dataset, Encoder, Row}
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.ml.attribute.AttributeGroup
import org.apache.spark.mllib.stat.MultivariateOnlineSummarizer
import org.apache.spark.mllib
import org.apache.spark.ml.param.{BooleanParam, Param, ParamMap}
import org.apache.spark.sql.types._


trait LinearRegressionParams extends HasFeaturesCol with HasLabelCol with HasPredictionCol {
  def setFeaturesCol(value: String) : this.type = set(featuresCol, value)
  def setLabelCol(value: String): this.type = set(labelCol, value)
  def setPredictionCol(value: String): this.type = set(predictionCol, value)

  protected def validateAndTransformSchema(schema: StructType): StructType = {
    SchemaUtils.checkColumnType(schema, getFeaturesCol, new VectorUDT())
    SchemaUtils.checkColumnType(schema, getLabelCol, DoubleType)

    if (schema.fieldNames.contains($(predictionCol))) {
      SchemaUtils.checkColumnType(schema, getPredictionCol, DoubleType)
      schema
    } else {
      SchemaUtils.appendColumn(schema, schema(getLabelCol).copy(name = getPredictionCol))
    }
  }
}

class LinearRegression(override val uid: String) extends Estimator[LinearRegressionModel] with LinearRegressionParams
with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("linearRegression"))

  override def fit(dataset: Dataset[_]): LinearRegressionModel = {

    val numIterations: Int = 2000
    val lr: Double = 0.001

    val numFeatures = dataset.select(dataset($(featuresCol))).first.getAs[Vector](0).size
    var weights = Vectors.dense(Array.fill(numFeatures)(0.0))
    var bias = 1.0
    
    // Perform gradient descent for the specified number of iterations
    for (i <- 0 until numIterations) {
        val gradient = dataset.select(dataset($(featuresCol)), dataset($(labelCol))).rdd.map { row =>
        val features = row.getAs[Vector](0)
        val label = row.getDouble(1)
        val prediction = weights.dot(features) + bias
        val error = prediction - label
        val weightGradient = features.toArray.map(_ * error)
        val biasGradient = error
        (Vectors.dense(weightGradient), biasGradient)
        }.reduce { case ((w1, b1), (w2, b2)) =>
        (Vectors.dense((w1.toArray, w2.toArray).zipped.map(_ + _)), b1 + b2)
        }
        
        // Update the weights and bias
        weights = Vectors.dense((weights.toArray, gradient._1.toArray).zipped.map(_ - lr * _))
        bias -= lr * gradient._2
    }

    copyValues(new LinearRegressionModel(
      weights, bias)).setParent(this)
  }

  override def copy(extra: ParamMap): Estimator[LinearRegressionModel] = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)

}

object LinearRegression extends DefaultParamsReadable[LinearRegression]

class LinearRegressionModel private[made](
                           override val uid: String,
                           val weights: DenseVector,
                           val bias: Double,
                           ) extends Model[LinearRegressionModel] with LinearRegressionParams with MLWritable {


  private[made] def this(weights: Vector, bias: Double) =
    this(Identifiable.randomUID("linearRegressionModel"), weights.toDense, bias)

  override def copy(extra: ParamMap): LinearRegressionModel = copyValues(
    new LinearRegressionModel(weights, bias), extra)

  override def transform(dataset: Dataset[_]): DataFrame = {
    val transformUdf = 
      dataset.sqlContext.udf.register(uid + "_transform",
        (x : Vector) => {(weights.asBreeze dot x.asBreeze) + bias})

    dataset.withColumn($(predictionCol), transformUdf(dataset($(featuresCol))))
  }

  override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)

  override def write: MLWriter = new DefaultParamsWriter(this) {
    override protected def saveImpl(path: String): Unit = {
      super.saveImpl(path)

      val vectors: (Vector, Double) = weights.asInstanceOf[Vector] -> bias.asInstanceOf[Double]

      sqlContext.createDataFrame(Seq(vectors)).write.parquet(path + "/vectors")
    }
  }
}

object LinearRegressionModel extends MLReadable[LinearRegressionModel] {
  override def read: MLReader[LinearRegressionModel] = new MLReader[LinearRegressionModel] {
    override def load(path: String): LinearRegressionModel = {
      val metadata = DefaultParamsReader.loadMetadata(path, sc)

      val params = sqlContext.read.parquet(path + "/vectors")

      // Used to convert untyped dataframes to datasets with vectors
      implicit val encoder : Encoder[Vector] = ExpressionEncoder()
      implicit val encoder2 : Encoder[Double] = ExpressionEncoder()

      val (weights, bias) =  params.select(params("_1").as[Vector], params("_2").as[Double]).first()

      val model = new LinearRegressionModel(weights, bias)
      metadata.getAndSetParams(model)
      model
    }
  }
}
