package com.example.wine_predictor;


import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.classification.LogisticRegressionModel;
import org.apache.spark.mllib.evaluation.MulticlassMetrics;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import scala.Tuple2;

public class PredictionModel {
	 public static void main(String[] args) {
	        // Initialize Spark Session and Context
	        SparkSession spark = SparkSession.builder()
	                .appName("Wine Quality Prediction - Predict Model")
	                .getOrCreate();
	        try (JavaSparkContext sc = new JavaSparkContext(spark.sparkContext())) {
	            // Load the Trained Model from S3
	            LogisticRegressionModel model = LogisticRegressionModel.load(sc.sc(), "\"C:\\Users\\ai057\\PA2\\ValidationDataset-1 (1).csv\""); 

	            // Load Test Dataset from S3
	            String testDataPath = "\"C:\\Users\\ai057\\PA2\\ValidationDataset-1 (1).csv\""; // Replace with actual test dataset path
	            Dataset<Row> testData = spark.read()
	                    .format("csv")
	                    .option("header", "true")
	                    .option("inferSchema", "true")
	                    .option("delimiter", ";") // Specify semicolon delimiter
	                    .load(testDataPath);

	            // Transform Test Data into LabeledPoint
	            JavaRDD<LabeledPoint> testRDD = testData.javaRDD()
	                    .map(row -> new LabeledPoint(
	                            row.getAs("\"quality\""),
	                            Vectors.dense(
	                                    row.getAs("\"fixed acidity\""),
	                                    row.getAs("\"volatile acidity\""),
	                                    row.getAs("\"citric acid\""),
	                                    row.getAs("\"residual sugar\""),
	                                    row.getAs("\"chlorides\""),
	                                    row.getAs("\"free sulfur dioxide\""),
	                                    row.getAs("\"total sulfur dioxide\""),
	                                    row.getAs("\"density\""),
	                                    row.getAs("\"pH\""),
	                                    row.getAs("\"sulphates\""),
	                                    row.getAs("\"alcohol\"")
	                            )
	                    ));

	            // Make Predictions and Pair with Labels
	            JavaRDD<Tuple2<Object, Object>> predictionsAndLabels = testRDD.map(point ->
	                    new Tuple2<>(model.predict(point.features()), point.label())
	            );

	            // Evaluate Predictions with MulticlassMetrics
	            MulticlassMetrics metrics = new MulticlassMetrics(predictionsAndLabels.rdd());
	            double f1Score = metrics.weightedFMeasure();
	            System.out.println("F1 Score: " + f1Score);
	        }

	        // Stop Spark Session
	        spark.stop();
	    }

}
