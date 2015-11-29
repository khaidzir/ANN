/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package ann;

import java.io.File;
import java.util.logging.Level;
import java.util.logging.Logger;
import weka.core.converters.ConverterUtils;
import weka.core.Instances;

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
            int[] nbLayers = {2, 2, 2};
            myANN.setNbLayers(nbLayers);
            myANN.setDeltaMSE(0.001);
            myANN.setLearningRate(0.1);
            myANN.setTopology(MyANN.MULTILAYER_PERCEPTRON);
            myANN.setLearningRule(MyANN.SIMPLE_PERCEPTRON);
            myANN.setActivationFunction(MyANN.SIGMOID_FUNCTION);
            myANN.setMaxIteration(1000);
            myANN.setTerminationCondition(MyANN.TERMINATE_MSE);
            myANN.setInitialWeight(wp.weight);
            
            myANN.buildClassifier(data);
            for (int i = 0; i < test.numInstances(); i++) {
                System.out.print("kelas: {");
                for(double d : myANN.distributionForInstance(test.instance(i))) {
                    System.out.print(d+", ");
                }
                System.out.println("}");
            }
        } catch (Exception ex) {
            Logger.getLogger(Main.class.getName()).log(Level.SEVERE, null, ex);
        }
        
    }
}
