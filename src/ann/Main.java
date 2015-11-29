/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package ann;

import java.io.File;
import java.util.Random;
import java.util.logging.Level;
import java.util.logging.Logger;
import weka.classifiers.Evaluation;
import weka.core.converters.ConverterUtils;
import weka.core.Instances;
import weka.filters.supervised.instance.Resample;

/**
 *
 * @author yusuf
 */
public class Main {
    
    public static void main(String[] args) {
        String trainPath = null;
        String testPath = null;
        String weights = null;
        String predictPath = null;
        char activationFunction = MyANN.SIGMOID_FUNCTION,
                terminateCondition = MyANN.TERMINATE_MAX_ITERATION,
                learningRule = MyANN.PERCEPTRON_TRAINING_RULE,
                topology = MyANN.ONE_PERCEPTRON;
        double deltaMSE = 0.01;
        int maxIteration = 500;
        double learningRate = 0.3;
        double momentum = 0.2;
        int nbHidden = 0;
        int[] hiddenConf = null;
        boolean isCV = false;
        int numFolds = 10;
        boolean isEvaluate = false;
            
        if (args.length < 1 || args.length % 2 == 0) {
            
            System.out.println("Usage: ANN [-I <path>] [-t O|M] [-r P|B|D] [-h <layer>]"
                    + "\n\t [-a N|G|T] [-L <rate>] [-m <momentum>] [-E D|I|B] [-d <mse>]"
                    + "\n\t [-i <iteration>] [-e <path>|<n>] [-p <path>] <trainDataPath>");
            System.out.println("");
            System.out.println("-a N|G|T \t set activation function for OnePerceptron");
                System.out.println("\t\t   N=SIGN, G=SIGMOID, T=STEP");
            System.out.println("-d <mse> \t set MSE = <mse> for terminate condition");
            System.out.println("-E D|I|B \t\t set terminate condition, D=by MSE, I=by iteration");
            System.out.println("-e <path>|<n> \t set test data using <path> or cross-validation w/ folds = <n>");
            System.out.println("-h <layer> \t set hidden layer. <layer>=0 no hidden layer");
                System.out.println("\t\t   <layer>=2 => 1 hidden layer with 2 nodes");
                System.out.println("\t\t   <layer>=2,3 => 2 hidden layer with 2 nodes on first and 3 on second layer");
            System.out.println("-I <path> \t set initial weight from <path>");
            System.out.println("-i <iteration> \t set max iteration for terminate condition");
            System.out.println("-L <rate> \t set learning rate = <rate>");
            System.out.println("-m <momentum> \t set momentum = <momentum>");
            System.out.println("-p <path> \t set data to predict");
            System.out.println("-r P|B|D \t set learning rule for OnePerceptron ");
                System.out.println("\t\t   P=Perceptron training rule,B=Batch, D=DeltaRule");
            System.out.println("-t O|M \t set topology, O=OnePerceptron, M=MLP");
            return;
        } else {    
            trainPath = args[args.length - 1];
            
            int i = 0;
            while (i < args.length - 1) {
                switch(args[i]) {
                    case "-a":
                        switch(args[i+1]) {
                            case "N":
                                activationFunction = MyANN.SIGN_FUNCTION;
                                break;
                            case "G":
                                activationFunction = MyANN.SIGMOID_FUNCTION;
                                break;
                            case "T":
                                activationFunction = MyANN.STEP_FUNCTION;
                                break;
                            default:
                                break;
                        }
                        break;
                    case "-d":
                        deltaMSE = Double.valueOf(args[i+1]);
                        break;
                    case "-E":
                        switch(args[i+1]) {
                            case "D":
                                terminateCondition = MyANN.TERMINATE_MSE;
                                break;
                            case "I":
                                terminateCondition = MyANN.TERMINATE_MAX_ITERATION;
                                break;
                            case "B":
                                terminateCondition = MyANN.TERMINATE_BOTH;
                            default:
                                break;
                        }
                        break;
                    case "-e":
                        if (args[i+1].length() <= 2) {
                            numFolds = Integer.parseInt(args[i+1]);
                            isCV = true;
                        } else {
                            isEvaluate = true;
                            testPath = args[i+1];
                        }
                        break;
                    case "-h":
                        String[] nbl = args[i+1].split(",");
                        if (nbl.length == 1) {
                            nbHidden = Integer.parseInt(nbl[0]);
                            if (nbHidden != 0) {
                                hiddenConf = new int[1];
                                hiddenConf[0] = nbHidden;
                                nbHidden = 1;
                            }
                        } else {
                            nbHidden = nbl.length;
                            hiddenConf = new int[nbHidden];
                            for(int j = 0; j < nbHidden; j++) {
                                hiddenConf[j] = Integer.parseInt(nbl[j]);
                            }
                        }
                        break;
                    case "-I":
                        weights = args[i+1];
                        break;
                    case "-i":
                        maxIteration = Integer.parseInt(args[i+1]);
                        break;
                    case "-L":
                        learningRate = Double.parseDouble(args[i+1]);
                        break;
                    case "-m":
                        momentum = Double.parseDouble(args[i+1]);
                        break;
                    case "-p":
                        predictPath = args[i+1];
                        break;
                    case "-r":
                        switch(args[i+1]) {
                            case "P":
                                learningRule = MyANN.PERCEPTRON_TRAINING_RULE;
                                break;
                            case "B":
                                learningRule = MyANN.BATCH_GRADIENT_DESCENT;
                                break;
                            case "D":
                                learningRule = MyANN.DELTA_RULE;
                                break;
                            default:
                                break;
                        }
                        break;
                    case "-t":
                        switch(args[i+1]) {
                            case "O": 
                                topology = MyANN.ONE_PERCEPTRON;
                                break;
                            case "M":
                                topology = MyANN.MULTILAYER_PERCEPTRON;
                                break;
                            default:
                                break;
                        }
                        break;
                    default:
                        break;
                }
                i+=2;
            }
        }
        
        // persiapkan data
        Instances trainData = null;
        Instances testData = null;
        Instances predictData = null;
        try {
            ConverterUtils.DataSource source = new ConverterUtils.DataSource(trainPath);
            trainData = source.getDataSet();
            if (trainData.classIndex() == -1) {
                trainData.setClassIndex(trainData.numAttributes() - 1);
            }
            if (testPath != null) {
                source = new ConverterUtils.DataSource(testPath);
                testData = source.getDataSet();
                if (testData.classIndex() == -1) {
                    testData.setClassIndex(testData.numAttributes() - 1);
                }
            }
            if (predictPath != null) {
                source = new ConverterUtils.DataSource(predictPath);
                predictData = source.getDataSet();
                if (predictData.classIndex() == -1) {
                    predictData.setClassIndex(predictData.numAttributes() - 1);
                }
            }
        } catch (Exception ex) {
            Logger.getLogger(Main.class.getName()).log(Level.SEVERE, null, ex);
        }
        
        // persiapkan model dan parameter
        MyANN myAnn = new MyANN();
        WeightParser wp = null;
        if (weights != null) {
            wp = new WeightParser(weights);
            myAnn.setInitialWeight(wp.weight);
        }
        myAnn.setActivationFunction(activationFunction);
        myAnn.setDeltaMSE(deltaMSE);   
        myAnn.setLearningRate(learningRate);
        myAnn.setLearningRule(learningRule);
        myAnn.setMaxIteration(maxIteration);
        myAnn.setMomentum(momentum);
        myAnn.setTerminationCondition(terminateCondition);
        myAnn.setThreshold(momentum);
        myAnn.setTopology(topology);
        
        int[] nbLayer = new int[2];
        if(nbHidden != 0) {
            nbLayer = new int[2 + nbHidden];
            for (int j = 1; j < nbLayer.length - 1; j++) {
                nbLayer[j] = hiddenConf[j-1];
            }
        }
        nbLayer[0] = trainData.numAttributes() - 1;
        if (trainData.classAttribute().isNominal())
            nbLayer[nbLayer.length - 1] = trainData.classAttribute().numValues();
        else
            nbLayer[nbLayer.length - 1] = 1;
        
        myAnn.setNbLayers(nbLayer);
        
        // debug: cek kondigurasi
        System.out.println("training data: "+trainPath);
        System.out.println("settings:");
        myAnn.printSetting();
        System.out.println("");
        
        // klasifikasi
        System.out.println("start classifiying...");
        try {
            myAnn.buildClassifier(trainData);
        } catch (Exception ex) {
            Logger.getLogger(Main.class.getName()).log(Level.SEVERE, null, ex);
        }
        myAnn.printSummary();
        System.out.println("done");
        
        System.out.println("-------------------------------------------------");
        
        System.out.print("evaluating ");
        int[][] result = null;
        int nbData = trainData.numInstances();
        if (isCV) {
            System.out.println("using "+numFolds+"-folds cross validation");
            result = myAnn.crossValidation(trainData, numFolds, new Random(1));
        } else if (isEvaluate) {
            System.out.println("using testData: "+testPath);
            result = myAnn.evaluate(testData);
            nbData = testData.numInstances();
        } else {
            System.out.println("using trainData");
            result = myAnn.evaluate(trainData);
        }
        System.out.println("");
        
        System.out.println("result:");

        
        double accuracy = 0.0;      // a+d/total
        double[] precision = new double[result.length];     // a/a+c;   prec[i] = M[i,i] / sumj(M[j,i])
        double[] recall = new double[result[0].length];        // a/a+b;   rec[i] = M[i,i] / sumj(M[i,j])

        for (int i = 0; i < result.length; i++) {
            for (int j = 0; j < result[0].length; j++) {
                System.out.print(result[i][j] + " ");
                if (i==j) {
                    accuracy += result[i][j];
                }
            }
            System.out.println("");
        }

        // precision
        for(int i = 0; i < precision.length; i++) {
            double sum = 0.0;
            for (int j = 0; j < result[0].length; j++) {
                sum += result[j][i];
            }
            precision[i] = result[i][i] / sum;
        }

        // recall
        for(int i = 0; i < recall.length; i++) {
            double sum = 0.0;
            for (int j = 0; j < result[0].length; j++) {
                sum += result[i][j];
            }
            recall[i] = result[i][i] / sum;
        }

        accuracy /= nbData;
        System.out.println("");
        System.out.println("accuracy: "+accuracy);
        System.out.println("precision: ");
        for(double p : precision) {
            System.out.println(p);
        }
        System.out.println("");
        System.out.println("recall: ");
        for (double r : recall) System.out.println(r);
        System.out.println("");
        System.out.println("-------------------------------------------------");
        
        if (predictPath != null) {
            System.out.println("predicting: "+predictPath);
            for (int i = 0; i < predictData.numInstances(); i++) {
                try {
                    int idx = myAnn.predictClassIndex(myAnn.distributionForInstance(predictData.instance(i)));
                    System.out.println("instance["+(i)+"]: "+trainData.classAttribute().value(idx));
                } catch (Exception ex) {
                    Logger.getLogger(Main.class.getName()).log(Level.SEVERE, null, ex);
                }
            }
            System.out.println("done");
        }
        /*
        try {
            File file = new File("/media/yusuf/5652859E52858389/Data/Kuliah/Semester 7/ML/WekaMiddle/weather.nominal.arff");
            File unlabel = new File("/media/yusuf/5652859E52858389/Data/Kuliah/Semester 7/ML/WekaMiddle/weather.nominal.unlabeled.arff");
            Instances data, test;
            ConverterUtils.DataSource source = new ConverterUtils.DataSource(file.getPath());
            data = source.getDataSet();
            if (data.classIndex() == -1) {
                data.setClassIndex(data.numAttributes() - 1);
            }
            source = new ConverterUtils.DataSource(unlabel.getPath());
            test = source.getDataSet();
            if (test.classIndex() == -1) {
                test.setClassIndex(data.numAttributes() - 1);
            }
            
            WeightParser wp = new WeightParser("/media/yusuf/5652859E52858389/Data/Kuliah/Semester 7/ML/khaidzir_myANN/initial.weight");
            MyANN myANN = new MyANN();
            int[] nbLayers = {4, 3, 2};
            myANN.setNbLayers(nbLayers);
            myANN.setDeltaMSE(0.001);
            //myANN.setMomentum(0.2);
            myANN.setLearningRate(0.1);
            myANN.setTopology(MyANN.MULTILAYER_PERCEPTRON);
            myANN.setLearningRule(MyANN.PERCEPTRON_TRAINING_RULE);
            myANN.setActivationFunction(MyANN.SIGMOID_FUNCTION);
            myANN.setMaxIteration(10000);
            myANN.setTerminationCondition(MyANN.TERMINATE_MAX_ITERATION);
            //myANN.setInitialWeight(wp.weight);
            
            
            myANN.buildClassifier(data);
            int[][] ev = myANN.evaluate(data);
            for (int[] ev1 : ev) {
                for (int ev2 : ev1) {
                    System.out.print(ev2+", ");
                }
                System.out.println("");
            }
            System.out.println("");
            //ev = myANN.crossValidation(data, 10, new Random(1));
            for (int[] ev1 : ev) {
                for (int ev2 : ev1) {
                    System.out.print(ev2+", ");
                }
                System.out.println("");
            }
            System.out.println("");
            
            /*
            myANN.buildClassifier(data);
            int[][] cm = myANN.evaluate(data);
            double accuracy = 0.0;      // a+d/total
            double[] precision = new double[cm.length];     // a/a+c;   prec[i] = M[i,i] / sumj(M[j,i])
            double[] recall = new double[cm[0].length];        // a/a+b;   rec[i] = M[i,i] / sumj(M[i,j])
            
            for (int i = 0; i < cm.length; i++) {
                for (int j = 0; j < cm[0].length; j++) {
                    System.out.print(cm[i][j] + " ");
                    if (i==j) {
                        accuracy += cm[i][j];
                    }
                }
                System.out.println("");
            }
            
            // precision
            for(int i = 0; i < precision.length; i++) {
                double sum = 0.0;
                for (int j = 0; j < cm[0].length; j++) {
                    sum += cm[j][i];
                }
                precision[i] = cm[i][i] / sum;
            }
            
            // recall
            for(int i = 0; i < recall.length; i++) {
                double sum = 0.0;
                for (int j = 0; j < cm[0].length; j++) {
                    sum += cm[i][j];
                }
                recall[i] = cm[i][i] / sum;
            }
            
            accuracy /= data.numInstances();
            System.out.println("accuracy: "+accuracy);
            System.out.println("precision: ");
            for(double p : precision) {
                System.out.print(p+", ");
            }
            System.out.println("");
            System.out.println("recall: ");
            for (double r : recall) System.out.print(r+", ");
            System.out.println("");

        } catch (Exception ex) {
            Logger.getLogger(Main.class.getName()).log(Level.SEVERE, null, ex);
        }
        */
    }
}
