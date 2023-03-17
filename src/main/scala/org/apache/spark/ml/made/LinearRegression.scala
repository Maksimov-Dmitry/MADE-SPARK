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

    // Used to convert untyped dataframes to datasets with vectors
    implicit val encoder : Encoder[Vector] = ExpressionEncoder()
    implicit val encoder2 : Encoder[Double] = ExpressionEncoder()

    val vectors: Dataset[Vector] = dataset.select(dataset($(featuresCol)).as[Vector])
    val labels: Dataset[Double] = dataset.select(dataset($(labelCol)).as[Double])
    val max_interations: Int = 100
    val lr: Double = 0.001
    val n: Double = dataset.count()

    val dim: Int = AttributeGroup.fromStructField((dataset.schema($(featuresCol)))).numAttributes.getOrElse(
      vectors.first().size
    )
    val weights = Vectors.zeros(dim)
    var bias = 1.0

    var i = 0
    while (i < max_interations) {
      // Calculate gradients
      val gradients = dataset.rdd.map(row => {
        val features = row.getAs[Vector]($(featuresCol))
        val label = row.getAs[Double]($(labelCol))
        val prediction = weights.dot(features) + bias
        val error = prediction - label
        val gradientW = features * error
        val gradientB = error
        (gradientW, gradientB)
      }).reduce((a, b) => (a._1 + b._1, a._2 + b._2))
        weights.toArray.indices.foreach(j => weights.toArray.update(j, weights.toArray(j) - lr * gradients._1.toArray(j) / assembledData.count()))
        bias -= lr * gradients._2 / dataset.count()
        i += 1
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
