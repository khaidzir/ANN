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
            int[] nbLayers = {4, 2, 2};
            myANN.setNbLayers(nbLayers);
            myANN.setDeltaMSE(0.001);
            myANN.setLearningRate(0.1);
            myANN.setTopology(MyANN.MULTILAYER_PERCEPTRON);
            myANN.setLearningRule(MyANN.SIMPLE_PERCEPTRON);
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
*/
        } catch (Exception ex) {
            Logger.getLogger(Main.class.getName()).log(Level.SEVERE, null, ex);
        }
        
    }
}
