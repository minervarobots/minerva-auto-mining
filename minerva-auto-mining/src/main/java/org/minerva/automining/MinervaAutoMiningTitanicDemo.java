package org.minerva.automining;

import java.util.Random;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.trees.J48;
import weka.core.AttributeStats;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;
import weka.core.converters.ConverterUtils.DataSource;
import weka.experiment.Stats;

public class MinervaAutoMiningTitanicDemo {

    public static void main(String[] args) throws Exception {

    	String trainData = "dataset/kaggle-weka-dataset/titanic/train.arff";
    	String testData = "dataset/kaggle-weka-dataset/titanic/test.arff";
    	
//        ConverterUtils.DataSource source1 = new ConverterUtils.DataSource("dataset/weka-dataset-arff/iris.arff");  	
        ConverterUtils.DataSource source1 = new ConverterUtils.DataSource(trainData);
        Instances train = source1.getDataSet();
        
        
        // setting class attribute if the data format does not provide this information
        // For example, the XRFF format saves the class attribute information as well
        
        if (train.classIndex() == -1)
            train.setClassIndex(train.numAttributes() - 1);

//        ConverterUtils.DataSource source2 = new ConverterUtils.DataSource("dataset/weka-dataset-arff/iris-test.arff");
        ConverterUtils.DataSource source2 = new ConverterUtils.DataSource(testData);

        
        Instances test = source2.getDataSet();
        // setting class attribute if the data format does not provide this information
        // For example, the XRFF format saves the class attribute information as well
        
        
        if (test.classIndex() == -1)
            test.setClassIndex(train.numAttributes() - 1);

        // model

        NaiveBayes naiveBayes = new NaiveBayes();
        naiveBayes.buildClassifier(train);
        
        
        // Model 
        Classifier cls = new weka.classifiers.trees.J48();
        cls.buildClassifier(train);
        System.out.println(cls);
        
        
        // Statistics
        //Classifier cls = new J48();
        Evaluation eval = new Evaluation(train);
        Random rand = new Random(1234);  // using seed = 1
        int folds = 10;
        eval.crossValidateModel(cls, train, folds, rand);
        System.out.println(eval.toSummaryString());
        
        
        // Detailed class statistics
        
        //Classifier cls = new J48();
        //Evaluation eval = new Evaluation(data);
        //Random rand = new Random(1);  // using seed = 1
        //int folds = 10;
        eval.crossValidateModel(cls, train, folds, rand);
        System.out.println(eval.toClassDetailsString());
        
        
        // Confusion matrix
        
        // Generating classifier evaluation output manually
        // https://weka.wikispaces.com/Generating+classifier+evaluation+output+manually
        // print stats 
       
//        Classifier cls = new J48();
        //Evaluation eval = new Evaluation(train);
//        Random rand = new Random(1234);  // using seed = 1
//        int folds = 10;
        eval.crossValidateModel(cls, train, folds, rand);
        System.out.println(eval.toMatrixString());
        
        

        // this does the trick  
        double label = naiveBayes.classifyInstance(test.instance(0));
        
       
        // Manual Test
//        System.out.println("---------- Manual Text Print ------");        
//        test.instance(0).setClassValue(label);
//        test.instance(1).setClassValue(label);
//        test.instance(5).setClassValue(label);

        
        // Manual test print
//        System.out.println(test.instance(0).stringValue(4));
//        System.out.println(test.instance(1).stringValue(4));        
//        System.out.println(test.instance(5).stringValue(4));
//        System.out.println("---------- End Manual Text Print ------");        
        

//        System.out.println("---------- Loop Text Print ------");        
        System.out.println("Train Num Instances: " + train.numInstances());
        System.out.println("Test Num Instances: " + test.numInstances());

        
        // Loop 
//        for (int i = 0; i < test.numInstances(); i++) {
        	
//            test.instance(i).setClassValue(label);
            
//            System.out.println("-----------------------------");
//            System.out.println(test.instance(i).stringValue(4));
//            System.out.println(test.instance(i).value(2));
//            System.out.println(test.instance(i).value(3));
//            System.out.println(test.instance(i).stringValue(4));
//            System.out.println("-----------------------------");
//            System.out.println("|");

//         }
        
        
        // Variables Stats Train Dataset 
        
        //load dataset
//    	DataSource source = new DataSource("dataset/weka-dataset-arff/iris.arff");
    	DataSource source = new DataSource(trainData);
    	
    	
        //get instances object 
        Instances data = source.getDataSet();

        // Print Train Dataset Variables Stats
        System.out.println("---------- Print Train Dataset Variables Stats ------");        
        
        
        //set class index .. as the last attribute
        if (data.classIndex() == -1) {
           data.setClassIndex(data.numAttributes() - 1);
        }
        //get number of attributes (notice class is not counted)
        int numAttr = data.numAttributes() - 1;
        for (int i = 0; i < numAttr; i++) {
            //check if current attr is of type nominal
            if (data.attribute(i).isNominal()) {
                System.out.println("The "+i+"th Attribute is Nominal"); 
                //get number of values
                int n = data.attribute(i).numValues();
                System.out.println("The "+i+"th Attribute has: "+n+" values");
            }           

            //get an AttributeStats object
            AttributeStats as = data.attributeStats(i);
            int dC = as.distinctCount;
            System.out.println("The "+i+"th Attribute has: "+dC+" distinct values");

            //get a Stats object from the AttributeStats
            if (data.attribute(i).isNumeric()){
                System.out.println("The "+i+"th Attribute is Numeric"); 
                Stats s = as.numericStats;
                System.out.println("The "+i+"th Attribute has min value: "+s.min+" and max value: "+s.max+" and mean value: "+s.mean+" and stdDev value: "+s.stdDev );
            }

    }
        
        
    }
}