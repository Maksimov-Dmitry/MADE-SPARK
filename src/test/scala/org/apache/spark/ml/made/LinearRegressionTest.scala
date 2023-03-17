package org.apache.spark.ml.made

import com.google.common.io.Files
import org.scalatest._
import flatspec._
import matchers._
import org.apache.spark.ml
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions.lit
import breeze.linalg._
import org.apache.spark.sql.{DataFrame, Dataset, Encoder, Row}
import org.apache.spark.sql.functions.udf
import breeze.numerics.abs


class LinearRegressionTest extends AnyFlatSpec with should.Matchers with WithSpark {

  lazy val data: DataFrame = LinearRegressionTest._df
  lazy val expected_weights = LinearRegressionTest._true_weights
  lazy val expected_bias: Double = LinearRegressionTest._true_bias
  val delta = 0.1

  "Model" should "predict" in {
    val model: LinearRegressionModel = new LinearRegressionModel(
      weights = Vectors.dense(0.0, 0.0, 0.0).toDense,
      bias = 2.0
    ).setFeaturesCol("features").setLabelCol("target")

    val vectors = model.transform(data)
    vectors.count() should be(100000)
  }

  "Estimator" should "calculate weights and bias" in {
    val estimator = new LinearRegression()
      .setFeaturesCol("features")
      .setLabelCol("target")
      .setPredictionCol("prediction")

    val model = estimator.fit(data)
    assert(max(abs(model.weights.asBreeze - expected_weights)) < delta)
    model.bias should be(expected_bias +- delta)

  }

  "Estimator" should "work after re-read" in {

    val pipeline = new Pipeline().setStages(Array(
      new LinearRegression()
        .setFeaturesCol("features")
        .setLabelCol("target")
        .setPredictionCol("prediction")
    ))

    val tmpFolder = Files.createTempDir()

    pipeline.write.overwrite().save(tmpFolder.getAbsolutePath)

    val reRead = Pipeline.load(tmpFolder.getAbsolutePath)

    val model = reRead.fit(data).stages(0).asInstanceOf[LinearRegressionModel]
    assert(max(abs(model.weights.asBreeze - expected_weights)) < delta)
    model.bias should be(expected_bias +- delta)
  }

  "Model" should "work after re-read" in {

    val pipeline = new Pipeline().setStages(Array(
      new LinearRegression()
        .setFeaturesCol("features")
        .setLabelCol("target")
        .setPredictionCol("prediction")
    ))

    val model = pipeline.fit(data)

    val tmpFolder = Files.createTempDir()

    model.write.overwrite().save(tmpFolder.getAbsolutePath)

    val reRead: PipelineModel = PipelineModel.load(tmpFolder.getAbsolutePath)
  }
}

object LinearRegressionTest extends WithSpark {
  lazy val _true_weights = Vectors.dense(1.5, 0.3, -0.7).asBreeze
  lazy val _true_bias: Double = 3.0
  lazy val _features = Seq.fill(100000)(
    Vectors.fromBreeze(DenseVector.rand(3))
  )

  lazy val _data: DataFrame = {
    import sqlc.implicits._
    _features.map(x => Tuple1(x)).toDF("features")
  }
  lazy val transformUdf = 
      udf((x : Vector) => {_true_weights dot x.asBreeze + _true_bias})

  lazy val _df = _data.withColumn("target", transformUdf(_data("features")))
}
