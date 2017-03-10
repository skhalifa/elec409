package ca.queenus;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaSparkContext;


import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.classification.LogisticRegressionModel;
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS;
import org.apache.spark.mllib.evaluation.MulticlassMetrics;
import org.apache.spark.mllib.linalg.Matrix;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.linalg.distributed.RowMatrix;
import org.apache.spark.mllib.stat.MultivariateStatisticalSummary;
import org.apache.spark.mllib.stat.Statistics;

import scala.Tuple2;



public class MyFirstMllib {
	
  
  public static void main(String[] args) {

    SparkConf conf = new SparkConf().setAppName("JavaSummaryStatisticsExample");
    JavaSparkContext jsc = new JavaSparkContext(conf);


    //read the iris.data.csv file from HDFS
    JavaRDD<String> csvFile = jsc.textFile("hdfs://bi-hadoop-prod-4132.bi.services.us-south.bluemix.net:8020/tmp/iris.data.csv");

    
    //Convert the CSV data to dense vectors RDDs so that we can use it in Spark.
    JavaRDD<Vector> mat = csvFile.map(new Function<String, Vector>(){

		private static final long serialVersionUID = 1L;

		@Override
		public Vector call(String arg0) throws Exception {
//			System.out.println("arg0"+arg0);
            String[] attributes = arg0.split(",");
            
            double[] values = new double[attributes.length];
            for (int i = 0; i < attributes.length; i++) {
        		values[i] = Double.parseDouble(attributes[i]);
//        		System.out.println(values[i]);
            }
            return Vectors.dense(values);  
		}
    	
    });
    
        
    // Compute column summary statistics.
    MultivariateStatisticalSummary summary = Statistics.colStats(mat.rdd());
    System.out.println("Mean:"+summary.mean());  // a dense vector containing the mean value for each column
    System.out.println("Variance:"+ summary.variance());  // column-wise variance
    System.out.println("Number of Non Zeros:"+summary.numNonzeros());  // number of nonzeros in each column
    
    // Compute the Pearson Correlation Matrix
    Matrix correlMatrix = Statistics.corr(mat.rdd(), "pearson");
    System.out.println(correlMatrix.toString());
    
    // Create a sample without replacement 0f 30% of the records
    boolean withReplacement = false;
    double fraction = 0.3;
    JavaRDD<Vector> sampleMat =  mat.sample(withReplacement, fraction);
    System.out.println("Sample"+sampleMat.toString());
    
    // Compute the top 3 principal components.
    RowMatrix rowmat = new RowMatrix(mat.rdd());
    Matrix pc = rowmat.computePrincipalComponents(3);
    System.out.println("PCA Matrix:"+pc.toString());
//    RowMatrix projected = rowmat.multiply(pc);
//    System.out.println("PCA Projected:"+projected.toString());
    
    
    //Convert the CSV data to LabeledPoint  RDDs so that we can use it in Spark Classification.
    JavaRDD<LabeledPoint> labeledmat = csvFile.map(new Function<String, LabeledPoint>(){

		private static final long serialVersionUID = 1L;

		@Override
		public LabeledPoint call(String arg0) throws Exception {
//			System.out.println("arg0"+arg0);
            String[] attributes = arg0.split(",");
            
            double[] values = new double[attributes.length];
            for (int i = 0; i < attributes.length-1; i++) {
        		values[i] = Double.parseDouble(attributes[i]);
//        		System.out.println(values[i]);
            }
            return new LabeledPoint(Double.parseDouble(attributes[attributes.length-1]), Vectors.dense(values));  
		}
    	
    });
    

    // Split initial RDD into two... [70% training data, 30% testing data].
    JavaRDD<LabeledPoint>[] splits = labeledmat.randomSplit(new double[] {0.7, 0.3}, 11L);
    JavaRDD<LabeledPoint> training = splits[0].cache();
    JavaRDD<LabeledPoint> test = splits[1];
    
   

    // Run training algorithm to build the model.
    LogisticRegressionModel model = new LogisticRegressionWithLBFGS().setNumClasses(4).run(training.rdd());

    // Compute raw scores on the test set.
    JavaPairRDD<Object, Object> predictionAndLabels = test.mapToPair(record -> new Tuple2<>(model.predict(record.features()), record.label()));

    // Get evaluation metrics.
    MulticlassMetrics metrics = new MulticlassMetrics(predictionAndLabels.rdd());
    double accuracy = metrics.accuracy();
    System.out.println("\n\n\n\n\n\nAccuracy = " + accuracy);

    // Save and load model
    model.save(jsc.sc(), "hdfs://bi-hadoop-prod-4132.bi.services.us-south.bluemix.net:8020/tmp/Model");
    System.out.println("Model Saved Successfully!!!"+model.toString()); 
    LogisticRegressionModel sameModel = LogisticRegressionModel.load(jsc.sc(),"hdfs://bi-hadoop-prod-4132.bi.services.us-south.bluemix.net:8020/tmp/Model");
    System.out.println("Model Loaded Successfully!!!"+sameModel.toString()); 
  
    jsc.stop();
    jsc.close();
  }
}
