/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package ann;

import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author YusufR
 */
public class myANN extends Classifier{
    
    /**
     * Daftar konstanta untuk berbagai parameter
     */
    // topologi apa yang digunakan, 1perceptron atau MLP
    public static final char ONE_PERCEPTRON = 1;
    public static final char MULTILAYER_PERCEPTRON = 2;
    
    // learning rule apa yang digunakan, 1perceptron, batch, atau delta rule
    public static final char SIMPLE_PERCEPTRON = 3;
    public static final char BATCH_GRADIENT_DESCENT = 4;
    public static final char DELTA_RULE = 5;
    
    // fungsi aktivasi yang digunakan step, sign, atau sigmoid
    public static final char STEP_FUNCTION = 6;
    public static final char SIGN_FUNCTION = 7;
    public static final char SIGMOID_FUNCTION = 8;
    
    // apakah menggunakan momentum atau tidak
    public static final char ENABLE_MOMENTUM = 9;
    public static final char DISABLE_MOMENTUM = 10;
    
    // jenis terminasi yang digunakan, deltaMSE atau maxIteration
    public static final char TERMINATE_MSE = 11;
    public static final char TERMINATE_MAX_ITERATION = 12;
    
    /**
     * Daftar variabel yang digunakan untuk perhitungan
     */
    private double learningRate;
    private double momentum;
    private double deltaMSE;    // input deltaMSE untuk terminasi
    private int maxIteration;   // input maxIteration untuk terminasi
    private double threshold;   // input threshold untuk fungsi aktivasi sign dan step

        /**
     * @return the learningRate
     */
    public double getLearningRate() {
        return learningRate;
    }

    /**
     * @param learningRate the learningRate to set
     */
    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }

    /**
     * @return the momentum
     */
    public double getMomentum() {
        return momentum;
    }

    /**
     * @param momentum the momentum to set
     */
    public void setMomentum(double momentum) {
        this.momentum = momentum;
    }

    /**
     * @return the deltaMSE
     */
    public double getDeltaMSE() {
        return deltaMSE;
    }

    /**
     * @param deltaMSE the deltaMSE to set
     */
    public void setDeltaMSE(double deltaMSE) {
        this.deltaMSE = deltaMSE;
    }

    /**
     * @return the maxIteration
     */
    public int getMaxIteration() {
        return maxIteration;
    }

    /**
     * @param maxIteration the maxIteration to set
     */
    public void setMaxIteration(int maxIteration) {
        this.maxIteration = maxIteration;
    }

    /**
     * @return the threshold
     */
    public double getThreshold() {
        return threshold;
    }

    /**
     * @param threshold the threshold to set
     */
    public void setThreshold(double threshold) {
        this.threshold = threshold;
    }
    
    /**
     * Mengembalikan default capability dari ANN
     * @return ANN's capability
     */
    @Override
    public Capabilities getCapabilities() { 
        Capabilities result = super.getCapabilities();
        result.disableAll();
        
        //attributes
        result.enable(Capability.NOMINAL_ATTRIBUTES);
        result.enable(Capability.NUMERIC_ATTRIBUTES);
        result.enable(Capability.DATE_ATTRIBUTES);
        result.enable(Capability.MISSING_VALUES);
        
        //class
        result.enable(Capability.NOMINAL_CLASS);
        result.enable(Capability.NUMERIC_CLASS);
        result.enable(Capability.DATE_CLASS);
        result.enable(Capability.MISSING_CLASS_VALUES);
        
        return result;
    }
    
    /**
     * Melakukan training dengan data yang diberikan
     * @param data training data
     * @throws Exception Exception apapun yang menyebabkan training gagal
     */
    @Override
    public void buildClassifier(Instances data) throws Exception {
        // TODO
    }
    
    /**
     * Fungsi untuk prediksi kelas dari suatu datum setelah model
     * selesai dibangun
     * @param i datum yang mau diklasifikasi
     * @return array of double berisi probabilitas masing-masing kelas
     * @throws Exception Exception apapun yang menyebabkan gagal memprediksi
     */
    @Override
    public double[] distributionForInstance(Instance i) throws Exception {
        // TODO
        double[] retArray = null;
        return retArray;
    }
    
    /**
     * Mengembalikan nilai 1 jika input di atas atau sama dengan threshold
     * dan mengembalikan nilai -1 jika di bawah threshold
     * @param _value nilai input
     * @return nilai aktivasi
     */
    private double sign(double _value) {
        double retval;
        if (_value < getThreshold()) {
            retval = 1.0;
        } else {
            retval = -1.0;
        }
        return retval;
    }
    
    /**
     * Mengembalikan nilai 1 jika input di atas atau sama dengan threshold
     * dan mengembalikan nilai 0 jika di bawah threshold
     * @param _value nilai input
     * @return nilai aktivasi
     */
    private double step(double _value) {
        double retval;
        if (_value < getThreshold()) {
            retval = 1.0;
        } else {
            retval = 0.0;
        }
        return retval;
    }
    
    /**
     * Mengembalikan nilai aktivasi dengan rumus sigmoid
     * sigmoid(x) = 1 / (1+e^-x)
     * @param _value nilai input
     * @return nilai aktivasi
     */
    private double sigmoid(double _value) {
        return 1.0/(1.0+Math.exp(-_value));
    }


}
