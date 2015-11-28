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
    
    // jenis terminasi yang digunakan, deltaMSE atau maxIteration
    public static final char TERMINATE_MSE = 9;
    public static final char TERMINATE_MAX_ITERATION = 10;
    
    /**
     * Daftar variabel yang digunakan untuk perhitungan dan parameter
     */
    private double learningRate = 0.3;
    private double momentum = 0.2;
    private double deltaMSE = 0.01;    // input deltaMSE untuk terminasi
    private int maxIteration = 500;   // input maxIteration untuk terminasi
    private double threshold = 0.0;   // input threshold untuk fungsi aktivasi sign dan step
    private char topology = ONE_PERCEPTRON;
    private char learningRule = SIMPLE_PERCEPTRON;
    private char activationFunction = SIGMOID_FUNCTION;
    private char terminationCondition = TERMINATE_MSE;
    private boolean isInitialWeightSet = false;

    ////////////////////////////////
    ////     Setter-Getter      ////
    ////    untuk parameter     ////
    ////////////////////////////////
    
    /**
     * mengatur nilai weight awal 
     * @param _weight nilai weight awal
     */
    public void setInitialWeight(double[] _weight) {
        isInitialWeightSet = true;
        // TODO : set masing2 weight
    }

    /**
     * mendapatkan nilai learningRate
     * @return the learningRate
     */
    public double getLearningRate() {
        return learningRate;
    }

    /**
     * mengatur nilai learningRate
     * default = 0.3
     * @param learningRate the learningRate to set
     */
    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }

    /**
     * mendapatkan topologi yang sedang digunakan
     * @return the topology
     */
    public char getTopology() {
        return topology;
    }

    /**
     * mengatur topologi yang akan digunakan
     * (default) ONE_PERCEPTRON untuk topologi 1perceptron
     * MULTILAYER_PERCEPTRON untuk topologi MLP
     * @param topology the topology to set
     */
    public void setTopology(char topology) {
        this.topology = topology;
    }

    /**
     * mendapatkan info learning rule yang digunakan
     * @return the learningRule
     */
    public char getLearningRule() {
        return learningRule;
    }

    /**
     * mengatur learning rule yang akan digunakan
     * untuk topologi 1perceptron
     * (default) SIMPLE_PERCEPTRON = 1perceptron biasa
     * BATCH_GRADIENT_DESCENT = batch gradient descent
     * DELTA_RULE = delta rule
     * @param learningRule the learningRule to set
     */
    public void setLearningRule(char learningRule) {
        this.learningRule = learningRule;
    }

    /**
     * mendapatkan info fungsi aktivasi per neuron yang digunakan
     * @return the activationFunction
     */
    public char getActivationFunction() {
        return activationFunction;
    }

    /**
     * mengatur fungsi aktivasi untuk setiap neuron
     * (default) SIGMOID_FUNCTION = fungsi sigmoid
     * STEP_FUNCTION = fungsi step (0, 1)
     * SIGN_FUNCTION = fungsi sign (-1, 1)
     * @param activationFunction the activationFunction to set
     */
    public void setActivationFunction(char activationFunction) {
        this.activationFunction = activationFunction;
    }

    /**
     * mendapatkan info terminasi yang digunakan
     * @return the terminationCondition
     */
    public char getTerminationCondition() {
        return terminationCondition;
    }

    /**
     * mengatur kondisi terminasi
     * (default) TERMINATE_MSE = terminasi menggunakan MSE, gunakan dengan method setDeltaMSE(double)
     * TERMINATE_MAX_ITERATION = terminasi berdasarkan jumlah iterasi, gunakan dengan method setMaxIteration(int)
     * @param terminationCondition the terminationCondition to set
     */
    public void setTerminationCondition(char terminationCondition) {
        this.terminationCondition = terminationCondition;
    }

    /**
     * mendapatkan besar momentum
     * @return the momentum
     */
    public double getMomentum() {
        return momentum;
    }

    /**
     * mengatur besa momentum, default 0.2
     * @param momentum the momentum to set
     */
    public void setMomentum(double momentum) {
        this.momentum = momentum;
    }

    /**
     * mendapatkan nilai error minimal untuk terminasi
     * @return the deltaMSE
     */
    public double getDeltaMSE() {
        return deltaMSE;
    }

    /**
     * mengatur nilai error minimal untuk terminasi, default 0.01
     * terminationCondition = TEMRINATE_MSE
     * @param deltaMSE the deltaMSE to set
     */
    public void setDeltaMSE(double deltaMSE) {
        this.deltaMSE = deltaMSE;
    }

    /**
     * mendapatkan nilai jumlah iterasi maksimal
     * @return the maxIteration
     */
    public int getMaxIteration() {
        return maxIteration;
    }

    /**
     * mengatur nilai maksimal iterasi, default 500
     * @param maxIteration the maxIteration to set
     */
    public void setMaxIteration(int maxIteration) {
        this.maxIteration = maxIteration;
    }

    /**
     * mendapatkan nilai threshold yang digunakan untuk fungsi aktivasi
     * @return the threshold
     */
    public double getThreshold() {
        return threshold;
    }

    /**
     * mengatur nilai threshold yang digunakan untuk 
     * fungsi aktivasi STEP dan SIGN, default 0.0
     * @param threshold the threshold to set
     */
    public void setThreshold(double threshold) {
        this.threshold = threshold;
    }
 
    ////////////////////////////////////
    ////       Fungsi utama         ////
    ////  Klasifikasi dan Prediksi  ////
    ////////////////////////////////////
    
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
    
    ///////////////////////////////////////////////////////
    ////    Fungsi yang digunakan untuk perhitungan    ////
    ///////////////////////////////////////////////////////
    
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