package ca.queensu.wordcount;

import java.io.IOException;
import java.util.Iterator;

import org.apache.hadoop.io.*;
import org.apache.hadoop.mapred.*;

public class WordCountReduce extends MapReduceBase implements Reducer<Text, IntWritable, Text, IntWritable> {

public void reduce(Text key, Iterator<IntWritable> values, OutputCollector<Text, IntWritable> output, Reporter reporter) throws IOException {
	       int wordCount = 0;
	       while (values.hasNext()) {
	 		wordCount += values.next().get();
	       }
	       output.collect(key, new IntWritable(wordCount ));
	}
}
