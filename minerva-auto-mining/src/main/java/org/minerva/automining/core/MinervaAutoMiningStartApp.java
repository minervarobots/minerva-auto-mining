package org.minerva.automining.core;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.sql.ResultSet;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.meta.AutoWEKAClassifier;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;

//import org.minerva.automining.examples.AutoWEKAClassifierTest;

public class MinervaAutoMiningStartApp {

	public static void main(String[] args) throws Exception {
		// TODO Auto-generated method stub


		// Config 
		
    	String trainData = "dataset/kaggle-weka-dataset/titanic/train.arff";
    	String testData = "dataset/kaggle-weka-dataset/titanic/test.arff";
    	String WekaModelSavedFile = "savedmodels/j48.model";    	
    	String AutoWekaModelSavedFile = "savedmodels/autoweka.model";    	
		
		
		// Run a PCA on Train Data
    	PCA runPCA = new PCA();
    	runPCA.RunPCA(trainData);

    	// Load Dataset 
        ConverterUtils.DataSource source1 = new ConverterUtils.DataSource(trainData);
        Instances train = source1.getDataSet();
        
        
        // setting class attribute if the data format does not provide this information
        // For example, the XRFF format saves the class attribute information as well
        
        if (train.classIndex() == -1)
            train.setClassIndex(train.numAttributes() - 1);

        ConverterUtils.DataSource source2 = new ConverterUtils.DataSource(testData);

        
        Instances test = source2.getDataSet();
        // setting class attribute if the data format does not provide this information
        // For example, the XRFF format saves the class attribute information as well
        
        
        if (test.classIndex() == -1)
            test.setClassIndex(train.numAttributes() - 1);

        // Run NaiveBayes
        NaiveBayes naiveBayes = new NaiveBayes();
        naiveBayes.buildClassifier(train);
        System.out.println("------------------------------");                
        System.out.println("Print naiveBayes results: ");
        System.out.println("------------------------------");                        
        System.out.println(naiveBayes);
                
        
        // Run J48
        Classifier cls = new J48();
        cls.buildClassifier(train);
        System.out.println("------------------------------");                
        System.out.println("Print J48 results: ");
        System.out.println("------------------------------");                        
        System.out.println(cls);
    	
        // Run RandomForest 
        RandomForest rf = new RandomForest();
        rf.buildClassifier(train);
        System.out.println("------------------------------");                
        System.out.println("Print Random Forest results: ");        
        System.out.println("------------------------------");                
        System.out.println(rf);
        
        System.out.println("------------------------------");                
        
    	// Run Auto Weka        
        AutoWEKAClassifier autoweka = new AutoWEKAClassifier();
        autoweka.setTimeLimit(1);//in minutes
        autoweka.setnBestConfigs(1);
        autoweka.setParallelRuns(1);
        autoweka.setBatchSize("100");
        autoweka.setMemLimit(1024);//in MB
        autoweka.setDebug(true);
        autoweka.setSeed(123);
        autoweka.buildClassifier(train);
      
        // Print AutoWeka Model
        System.out.println(autoweka);
        
        
        // Test the dataset set again using the Winner Classifier
        Classifier winnerClassifier = AbstractClassifier.forName("weka.classifiers.trees.RandomForest", new String[]{"-I", "9", "-K", "19", "-depth", "9"});
        winnerClassifier.buildClassifier(train);
        
        System.out.println("winnerClassifier");
        System.out.println(winnerClassifier);
    	    	
		
    	
    	

	}

}
