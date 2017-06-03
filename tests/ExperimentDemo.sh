java -cp /Users/caiomsouza/git/github.com/minerva-auto-mining/minerva-auto-mining/target/minerva-auto-mining-0.0.1-SNAPSHOT.jar:/Users/caiomsouza/git/github.com/minerva-auto-mining/wekaexamples/lib/weka.jar org.minerva.automining.examples.ExperimentDemo 
   -classifier weka.classifiers.trees.J48
   -exptype classification
   -splittype crossvalidation
   -runs 10
   -folds 10
   -result /Users/caiomsouza/git/github.com/minerva-auto-mining/output/results.arff
   -t /Users/caiomsouza/git/github.com/minerva-auto-mining/minerva-auto-mining/dataset/weka-dataset-arff
/vote.arff
   -t /Users/caiomsouza/git/github.com/minerva-auto-mining/minerva-auto-mining/dataset/weka-dataset-arff
/iris.arff