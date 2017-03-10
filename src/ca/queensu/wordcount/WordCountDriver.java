package ca.queensu.wordcount;

import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapred.*;

public class WordCountDriver {

	public static void main(String[] args) throws Exception {
		  JobConf conf = new JobConf(WordCountDriver.class);
		  conf.setJobName("wordcount");

		  conf.setOutputKeyClass(Text.class);
		  conf.setOutputValueClass(IntWritable.class);

		  conf.setMapperClass(WordCountMap.class);
		  conf.setCombinerClass(WordCountReduce.class);
		  conf.setReducerClass(WordCountReduce.class);

		  conf.setInputFormat(TextInputFormat.class);
		  conf.setOutputFormat(TextOutputFormat.class);

		  FileInputFormat.setInputPaths(conf, new Path(args[0]));
		  FileOutputFormat.setOutputPath(conf, new Path(args[1]));

		  JobClient.runJob(conf);
		}	


}
