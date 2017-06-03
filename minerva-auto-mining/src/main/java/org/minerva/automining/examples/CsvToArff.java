package org.minerva.automining.examples;

import java.io.File;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;

public class CsvToArff {
    public static void main(String[] args) {
        try {
            //String f1 = "C://Users//user//Desktop//testfold//test.csv";
            //String f2 = "C://Users//user//Desktop//testfold//output.arff";
            
        	String f1 = "dataset/csv-dataset/kaggle-santander-train.csv";
            String f2 = "dataset/weka-dataset-arff/kaggle-santander-train2.arff";
            

            // load the CSV file (input file)
            CSVLoader loader = new CSVLoader();
            loader.setSource(new File(f1));
            String [] options = new String[1];
            options[0]="-H";
            loader.setOptions(options);

            Instances data = loader.getDataSet();
            System.out.println(data);

            // save as an  ARFF (output file)
            ArffSaver saver = new ArffSaver();
            saver.setInstances(data);
            saver.setFile(new File(f2));
            //saver.setDestination(new File(f2));
            saver.writeBatch();
        } catch(Exception e) {
        }
    }
}