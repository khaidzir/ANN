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
            Instances data;
            ConverterUtils.DataSource source = new ConverterUtils.DataSource(file.getPath());
            data = source.getDataSet();
            if (data.classIndex() == -1) {
                data.setClassIndex(data.numAttributes() - 1);
            }
            MyANN myANN = new MyANN();
            int[] nbLayers = {4, 2};
            myANN.setNbLayers(nbLayers);
            myANN.buildClassifier(data);
        } catch (Exception ex) {
            Logger.getLogger(Main.class.getName()).log(Level.SEVERE, null, ex);
        }
        
    }
}
