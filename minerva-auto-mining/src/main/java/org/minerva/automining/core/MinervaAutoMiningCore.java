/*
 *    This program is free software; you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation; either version 2 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program; if not, write to the Free Software
 *    Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */

/*
 *    
 *    Copyright (C) 2017 Caio Moreno
 *    
 *    This code will be improved, it is only a POC demo code.
 *    
 *
 */

// https://weka.wikispaces.com/Use+Weka+in+your+Java+code

// Kaggle Titanic competition with Weka
// http://www195.pair.com/mik3hall/weka_kaggle.html



// https://ianma.wordpress.com/2010/01/16/weka-with-java-eclipse-getting-started/


// https://weka.wikispaces.com/Use+WEKA+in+your+Java+code

// https://inclass.kaggle.com/c/utexasdatasciencelab/forums/t/24719/using-weka-as-a-data-mining-tool




package org.minerva.automining.core;

import java.io.FileOutputStream;
import java.io.ObjectOutputStream;
import java.util.Random;

//import autoweka.ClassifierResult.Metric;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.meta.AutoWEKAClassifier;
//import weka.classifiers.meta.AutoWEKAClassifier.Metric;
//import weka.classifiers.trees.J48;
import weka.core.AttributeStats;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;
import weka.core.converters.ConverterUtils.DataSource;
import weka.experiment.Stats;

public class MinervaAutoMiningCore {

    public static void main(String[] args) throws Exception {

    	String trainData = "dataset/kaggle-weka-dataset/titanic/train.arff";
    	String testData = "dataset/kaggle-weka-dataset/titanic/test.arff";
    	String WekaModelSavedFile = "savedmodels/j48.model";    	
    	String AutoWekaModelSavedFile = "savedmodels/autoweka.model";    	
    	
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
//        double label = naiveBayes.classifyInstance(test.instance(0));
        
       
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
     
        // Saving models http://weka.wikispaces.com/Saving+and+loading+models
        
        // Serialization
        // https://weka.wikispaces.com/Serialization
        
        
     // create J48
        //Classifier cls = new J48();
        
        // train
        //Instances inst = new Instances(new BufferedReader(new FileReader(trainData)));
        
//        train.setClassIndex(train.numAttributes() - 1);
//        cls.buildClassifier(train);
        
        
        
        // serialize model
        ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(WekaModelSavedFile));
        oos.writeObject(cls);
        oos.flush();
        oos.close();
        
        System.out.println("---------- Model Saved to:" + WekaModelSavedFile);        

        
        // Deserializing
        
     // deserialize model
        Classifier clsFromSavedModel = (Classifier) weka.core.SerializationHelper.read(WekaModelSavedFile);
        
        System.out.println("---------- Model Deserializing with success");  
        
        
        // ToDo 
        // 1. Running a knowledgeflow from Java
        // http://weka.sourceforge.net/doc.stable/weka/gui/beans/FlowRunner.html
        // https://github.com/bnjmn/weka/blob/master/weka/src/main/java/weka/gui/beans/FlowRunner.java
        // http://forums.pentaho.com/archive/index.php/t-76743.html
        
        // 2. Improve Generating cross-validation folds (Java approach)
        // https://weka.wikispaces.com/Generating+cross-validation+folds+%28Java+approach%29
        
        // 3. Variable importance in random forest
        // http://weka.8497.n7.nabble.com/Variable-importance-in-random-forest-td38317.html
        // http://weka.8497.n7.nabble.com/Find-importance-of-Variables-td13536.html
        // https://stats.stackexchange.com/questions/167828/weka-visualize-combined-trees-of-random-forest-classifier
        
        
        
        // 4. PCA
        // I have this code already 
        
        
        
        
        // 5. Running Auto-Weka from Java
        // https://automl.github.io/autoweka/weka/classifiers/meta/AutoWEKAClassifier.html
        // https://github.com/automl/autoweka/issues/31
        // Lars Kotthoff
        // https://www.linkedin.com/in/lars-kotthoff-8641856/?ppe=1
        // http://weka.8497.n7.nabble.com/Executing-Auto-WEKA-2-0-through-java-code-td37518.html
        
        // https://github.com/automl/autoweka
        
        AutoWEKAClassifier autoweka = new AutoWEKAClassifier();
//        autoweka.setTimeLimit(2);//in minutes

        autoweka.setTimeLimit(5);//in minutes
        //autoweka.setTimeLimit(15);// 15 in minutes

        autoweka.setnBestConfigs(1);
        autoweka.setParallelRuns(1);
//        autoweka.setBatchSize(100);
        
        //autoweka.setMetric(errorRate);;
        autoweka.setBatchSize("100");
//		autoweka.setMetric(metric errorRate);
        
//errorRate
// areaUnderROC
        
        //autoweka.setMetric(Metric.class.toString("errorRate"));;
      
        
        
        autoweka.setMemLimit(1024);//in MB
        autoweka.setDebug(true);
        autoweka.setSeed(123);
        autoweka.buildClassifier(data);
        
        // Print AutoWeka Model
        System.out.println(autoweka);
        
        System.out.println("Total Tried: " + autoweka.totalTried);

        // serialize model (Save Model to File)
        ObjectOutputStream oosAutoWekaModel = new ObjectOutputStream(new FileOutputStream(AutoWekaModelSavedFile));
        oosAutoWekaModel.writeObject(autoweka);
        oosAutoWekaModel.flush();
        oosAutoWekaModel.close();
        
        System.out.println("---------- Model Saved to:" + AutoWekaModelSavedFile);        
        
        
        
        // failed: Base learner must implement Randomizable!
        // https://github.com/automl/autoweka/issues/44
        // That warning is harmless. Auto-WEKA doesn't encode these requirements and you may occasionally see such warnings.
        
        
        // Error: Could not auto-detect the location of your Auto-WEKA install - have you moved the classes away from the 'params' diectory? 
        // http://weka.8497.n7.nabble.com/Auto-WEKA-Problem-td38858.html
        
        // It was fixed copying Auto-WEKA from ./wekafiles and putting into the project folder.
        // I added to class path the autoweka.jar 
        
        // Solution was here
        // http://weka.8497.n7.nabble.com/Auto-WEKA-Problem-td38858.html
        
        // 6. Add Example with Data base
        // https://gist.github.com/dexetra/1226623/502c6e78fea715c105c79163af34cdccf11fb056
        
        // 7. Add an example with Twitter
        // https://github.com/felipebravom
        // https://github.com/felipebravom/AffectiveTweets
        // http://www.cs.waikato.ac.nz/~fjb11/thesis.pdf
        // http://www.cs.waikato.ac.nz/~fjb11/
        
        
        
        // 8. Add correlation anaylsis
        
        
        
       
        
        
        
        
    }
}