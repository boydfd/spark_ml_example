import org.apache.spark.api.java.function.VoidFunction;
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator;
import org.apache.spark.ml.linalg.DenseVector;
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics;
import org.apache.spark.sql.*;
import org.apache.spark.ml.classification.MultilayerPerceptronClassificationModel;
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import scala.Function1;
import scala.Product2;
import scala.Tuple2;

import java.net.URL;
import java.util.Vector;
import java.util.function.Function;

import static org.apache.spark.sql.types.DataTypes.DoubleType;

public class Main {
    public static void main(String[] args) {

        SparkSession spark = SparkSession
                .builder()
                .appName("Application Name")
                .config("spark.master", "local")
                .getOrCreate();

        String path = Main.class.getClassLoader().getResource("iris_libsvm.txt").getPath();
        Dataset<Row> dataFrame = spark.read().format("libsvm").load(path);

        Dataset<Row>[] splits = dataFrame.randomSplit(new double[]{0.6, 0.4}, 1234L);
        Dataset<Row> train = splits[0];
        Dataset<Row> test = splits[1];

        int[] layers = new int[] {4, 5, 4, 3};

        MultilayerPerceptronClassifier trainer = new MultilayerPerceptronClassifier()
                .setLayers(layers)
                .setBlockSize(128)
                .setSeed(1234L)
                .setMaxIter(100);

        MultilayerPerceptronClassificationModel model = trainer.fit(train);

        Row a;
//        a.<Vector>getAs(0).get(0);
//        RowFactory.create(a.<DenseVector>getAs(0).values()[0], a.get(1));
        Dataset<Row> result = model.transform(test);
        Dataset<Row> predictionAndLabels = result.select("rawPrediction", "label")
                .map(row -> RowFactory.create(row.<DenseVector>getAs(0).values()[1], row.get(1)), Encoders.bean(Row.class));

//        val scoreAndLabels =
//                predictionAndLabels.rdd().map(
//                        row -> row.
//                ){
//            case Row(rawPrediction: Vector, label: Double) => (rawPrediction(1), label)
//            case Row(rawPrediction: Double, label: Double) => (rawPrediction, label)
//        }
        BinaryClassificationMetrics metrics = new BinaryClassificationMetrics(predictionAndLabels);
        System.out.println(metrics.roc().take(10));
        System.out.println(metrics.roc());

//        for (Tuple2<Object, Object> line : metrics.roc().collect()) {
//            System.out.println(line._1);
//            System.out.println(line._2);
//        }
        System.out.println(metrics.roc().count());

//        BinaryClassificationEvaluator evaluator = new BinaryClassificationEvaluator().setMetricName("areaUnderROC");
//
//        System.out.println("Test set accuracy = " + evaluator.evaluate(predictionAndLabels));
    }
}