package com.example.wine_predictor;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.classification.LogisticRegressionModel;
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS;
import org.apache.spark.mllib.evaluation.MulticlassMetrics;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import scala.Tuple2;


public class TrainingModel {
    public static void main(String[] args) {
        // Initialize Spark Session and Context
        SparkSession spark = SparkSession.builder()
                .appName("Wine Quality Prediction - Train Model")
                .getOrCreate();
        try (JavaSparkContext sc = new JavaSparkContext(spark.sparkContext())) {
            try {
                // Load Training Dataset from S3
                String trainingDataPath = "\"C:\\Users\\ai057\\PA2\\TrainingDataset-1 (1).csv\"";
                Dataset<Row> trainingData = spark.read()
                        .format("csv")
                        .option("header", "true")
                        .option("inferSchema", "true")
                        .option("delimiter", ";")
                        .load(trainingDataPath);

                // Transform Data into LabeledPoint for MLlib
                JavaRDD<LabeledPoint> trainingRDD = trainingData.javaRDD()
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

                // Train Logistic Regression Model
                LogisticRegressionModel model = new LogisticRegressionWithLBFGS()
                        .setNumClasses(10)
                        .run(trainingRDD.rdd());

                // Load Validation Dataset from S3
                String validationDataPath = "\\\"C:\\\\Users\\\\ai057\\\\PA2\\\\ValidationDataset-1 (1).csv\\\"";
                Dataset<Row> validationData = spark.read()
                        .format("csv")
                        .option("header", "true")
                        .option("inferSchema", "true")
                        .option("delimiter", ";")
                        .load(validationDataPath);

                // Transform Validation Data into LabeledPoint
                JavaRDD<LabeledPoint> validationRDD = validationData.javaRDD()
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

                // Evaluate Model with Validation Dataset
                JavaRDD<Tuple2<Object, Object>> predictionsAndLabels = validationRDD.map(point ->
                        new Tuple2<>(model.predict(point.features()), point.label())
                );

                MulticlassMetrics metrics = new MulticlassMetrics(predictionsAndLabels.rdd());
                double f1Score = metrics.weightedFMeasure();
                System.out.println("F1 Score: " + f1Score);

                // Save the Model to S3
                String modelPath = "\"C:\\Users\\ai057\\PA2\\TrainingDataset-1 (1).csv\"";
                model.save(sc.sc(), modelPath);

            } catch (Exception e) {
                System.err.println("Error occurred during training: " + e.getMessage());
                e.printStackTrace();
            }
        }

        // Stop Spark Session
        spark.stop();
    }
}


