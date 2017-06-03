package org.minerva.automining.examples;

import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;

public class WekaTestPredictWeather {

    public static void main(String[] args) throws Exception {

        ConverterUtils.DataSource source1 = new ConverterUtils.DataSource("dataset/weka-dataset-arff/weather-train.arff");
        Instances train = source1.getDataSet();
        
        
        // setting class attribute if the data format does not provide this information
        // For example, the XRFF format saves the class attribute information as well
        
        if (train.classIndex() == -1)
            train.setClassIndex(train.numAttributes() - 1);

        ConverterUtils.DataSource source2 = new ConverterUtils.DataSource("dataset/weka-dataset-arff/weather-test.arff");
        
        Instances test = source2.getDataSet();
        // setting class attribute if the data format does not provide this information
        // For example, the XRFF format saves the class attribute information as well
        
        
        if (test.classIndex() == -1)
            test.setClassIndex(train.numAttributes() - 1);

        // model

        NaiveBayes naiveBayes = new NaiveBayes();
        naiveBayes.buildClassifier(train);

        // this does the trick  
        double label = naiveBayes.classifyInstance(test.instance(0));
        
       
        // Manual Test
        System.out.println("---------- Manual Text Print ------");        
        test.instance(0).setClassValue(label);
        test.instance(1).setClassValue(label);
        test.instance(5).setClassValue(label);

        
        // Manual test print
        System.out.println(test.instance(0).stringValue(4));
        System.out.println(test.instance(1).stringValue(4));        
        System.out.println(test.instance(5).stringValue(4));
        System.out.println("---------- End Manual Text Print ------");        
        

        System.out.println("---------- Loop Text Print ------");        
        
        // Loop 
        for (int i = 0; i < test.numInstances(); i++) {
        	
            test.instance(i).setClassValue(label);
            
//            System.out.println("-----------------------------");
            System.out.println(test.instance(i).stringValue(4));
//            System.out.println(test.instance(i).value(2));
//            System.out.println(test.instance(i).value(3));
//            System.out.println(test.instance(i).stringValue(4));
//            System.out.println("-----------------------------");
//            System.out.println("|");

         }
        
        
        
    }
}