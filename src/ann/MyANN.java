/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package ann;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;
import java.util.logging.Level;
import java.util.logging.Logger;
import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NumericToBinary;

/**
 *
 * @author YusufR
 */
public class MyANN extends Classifier{
    
    /**
     * Daftar konstanta untuk berbagai parameter
     */
    // topologi apa yang digunakan, 1perceptron atau MLP
    public static final char ONE_PERCEPTRON = 1;
    public static final char MULTILAYER_PERCEPTRON = 2;
    
    // learning rule apa yang digunakan, 1perceptron, batch, atau delta rule
    public static final char PERCEPTRON_TRAINING_RULE = 3;
    public static final char BATCH_GRADIENT_DESCENT = 4;
    public static final char DELTA_RULE = 5;
    
    // fungsi aktivasi yang digunakan step, sign, atau sigmoid
    public static final char STEP_FUNCTION = 6;
    public static final char SIGN_FUNCTION = 7;
    public static final char SIGMOID_FUNCTION = 8;
    
    // jenis terminasi yang digunakan, deltaMSE atau maxIteration
    public static final char TERMINATE_MSE = 9;
    public static final char TERMINATE_MAX_ITERATION = 10;
    public static final char TERMINATE_BOTH = 11;
    
    /**
     * Daftar variabel yang digunakan untuk perhitungan dan parameter
     */
    private double learningRate = 0.3;
    private double momentum = 0.2;
    private double deltaMSE = 0.01;    // input deltaMSE untuk terminasi
    private int maxIteration = 500;   // input maxIteration untuk terminasi
    private double threshold = 0.0;   // input threshold untuk fungsi aktivasi sign dan step
    private char topology = ONE_PERCEPTRON;
    private char learningRule = PERCEPTRON_TRAINING_RULE;
    private char activationFunction = SIGMOID_FUNCTION;
    private char terminationCondition = TERMINATE_MSE;
    private boolean isInitialWeightSet = false;
    private int[] nbLayers;     // jumlah layer dan jumlah node setiap layer
    private double[][] weights; // weight awal, weights[0] untuk bobot neuron dan weights[1] untuk bobot bias
    private int iteration;      // jumalah iterasi klasifikasi
    
    private ArrayList<Data> datas;  // instances yang telah diubah ke dalam array of data
    private ANNModel annModel;      // model yang akan diklasifikasi dari training data dan akan digunakan untuk prediksi

    MyANN() {
        learningRate = 0.3;
        momentum = 0.2;
        deltaMSE = 0.01;   
        maxIteration = 500;
        threshold = 0.0;   
        topology = ONE_PERCEPTRON;
        learningRule = PERCEPTRON_TRAINING_RULE;
        activationFunction = SIGMOID_FUNCTION;
        terminationCondition = TERMINATE_MSE;
        isInitialWeightSet = false;
    }
    
    MyANN(MyANN c) {
        learningRate = c.learningRate;
        momentum = c.momentum;
        deltaMSE = c.deltaMSE;
        maxIteration = c.maxIteration;
        threshold = c.threshold;
        topology = c.topology;
        learningRule = c.learningRule;
        activationFunction = c.activationFunction;
        terminationCondition = c.terminationCondition;
        isInitialWeightSet = c.isInitialWeightSet;
        nbLayers = c.nbLayers;
        weights = c.weights;
        datas = c.datas;
        annModel = c.annModel;
    }

    ////////////////////////////////
    ////     Setter-Getter      ////
    ////    untuk parameter     ////
    ////////////////////////////////
    
    /**
     * mengatur nilai weight awal 
     * @param _weight nilai weight awal
     */
    public void setInitialWeight(double[][] _weight) {
        isInitialWeightSet = true;
        // TODO : set masing2 weight
        weights = _weight;
    }

    /**
     * mendapatkan jumlah masing-masing node setiap layer dan jumlah layer
     * @return the nbLayers
     */
    public int[] getNbLayers() {
        return nbLayers;
    }

    /**
     * mengatur jumlah masing-masing node setiap layer dan jumlah layer
     * @param nbLayers the nbLayers to set
     */
    public void setNbLayers(int[] nbLayers) {
        this.nbLayers = nbLayers;
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
 untuk topologi 1perceptron
 (default) PERCEPTRON_TRAINING_RULE = 1perceptron biasa
 BATCH_GRADIENT_DESCENT = batch gradient descent
 DELTA_RULE = delta rule
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
     * @param instances training data
     * @throws Exception Exception apapun yang menyebabkan training gagal
     */
    @Override
    public void buildClassifier(Instances instances) throws Exception {
        
        // cek apakah sesuai dengan data input
        getCapabilities().testWithFail(instances);
        // copy data dan buang semua missing class
        instances = new Instances(instances);
        instances.deleteWithMissingClass();
        
        // filter
        NumericToBinary ntb = new NumericToBinary();
        ntb.setInputFormat(instances);
        instances = Filter.useFilter(instances, ntb);
        
        // ubah instances ke data
        instancesToDatas(instances);
        
        // membangun ANN berdasarkan nbLayers
        // membuat layer
        ArrayList<ArrayList<Node>> layers = new ArrayList<>();
        for (int i = 0; i < nbLayers.length; i++) {
            layers.add(new ArrayList<>());
        }
        
        // inisialisasi bagian input layer
        for (int i = 0; i < nbLayers[0]; i++) {
            // set id, prevLayer = null, nextLayer = layers[1]
            layers.get(0).add(new Node("node-0"+"-"+i, null, layers.get(1)));
        }
        
        // inisialisasi bagian hidden layer
        for (int i = 1; i < nbLayers.length - 1; i++) {
            for (int j = 0; j < nbLayers[i]; j++) {
                // set id, prevLayer = layers[i-1], nextLayer = layers[i+1]
                layers.get(i).add(new Node("node-"+i+"-"+j, layers.get(i-1), layers.get(i+1)));
            }
        }
        
        // inisialisasi bagian output layer
        for (int i = 0; i < nbLayers[nbLayers.length - 1]; i++) {
            // set id, prevLayer = layers[n-1], nextLayer = null
            layers.get(nbLayers.length - 1).add(new Node("node-"+(nbLayers.length - 1)+"-"+i, layers.get(nbLayers.length - 2), null));
        }
        
        // tambah weight tiap neuron
        
        // siapin bobot bias, jumlah layer bias adalah nbLayers - 1
        ArrayList<Double> bias = new ArrayList<>();
        for (int i = 0; i < nbLayers.length - 1; i++) {
            bias.add(1.0);
        }
                
        // jumlah bobot setiap layer sama dengan jumlah node setiap layer                
        double[][] biasWeight = new double[nbLayers.length - 1][];
        for (int i = 1; i < biasWeight.length; i++) {
            biasWeight[i] = new double[nbLayers[i]];
        }
        
        // masukin setiap bobot dengan angka random
        Random rand = new Random(System.currentTimeMillis());
        // masukin bobot bias
        int j = 0;
        Map<Integer, Map<Node, Double>> biasesWeight = new HashMap<>();
        for (int i = 0; i < nbLayers.length - 1; i++) {
            ArrayList<Node> arrNode = layers.get(i+1);
            Map<Node, Double> map = new HashMap<>();
            for (Node node : arrNode) {
                if (isInitialWeightSet) {
                    map.put(node, weights[1][j]);
                } else {
                    map.put(node, rand.nextDouble());
                }
                j++;
            }
            biasesWeight.put(i, map);
        }
        
        j=0;
        // masukin bobot tiap neuron
        Map<Node, Map<Node, Double>> mapWeight = new HashMap<>();
        for (int i = 0; i < nbLayers.length-1; i++) {
            ArrayList<Node> arrNode = layers.get(i);
            for (Node node : arrNode) {
                Map<Node, Double> map = new HashMap<>();
                for (Node nextNode : node.getNextNodes()) {
                    if (isInitialWeightSet) {
                        map.put(nextNode, weights[0][j]);
                    } else {
                        map.put(nextNode, rand.nextDouble());
                    }
                    j++;
                }
                mapWeight.put(node, map);
            }
        }
        
        // buat model ANN berdasarkan nilai di atas
        annModel = new ANNModel(layers, mapWeight, bias, biasesWeight);
        // set konfigurasi awal model
        // debug
//        System.out.println("debug");
//        for (Data d : datas) {
//            for (Double dd : d.input) {
//                System.out.print(dd+" ");
//            }
//            System.out.print(" | ");
//            for (Double dd : d.target) {
//                System.out.print(dd+" ");
//            }
//            System.out.println("");
//        }
//        System.out.println("debug");
        annModel.setDataSet(datas);
        annModel.setLearningRate(learningRate);
        annModel.setMomentum(momentum);
        switch (activationFunction) {
            case SIGMOID_FUNCTION:
                annModel.setActivationFunction(ANNModel.SIGMOID);
                break;
            case SIGN_FUNCTION:
                annModel.setActivationFunction(ANNModel.SIGN);
                break;
            case STEP_FUNCTION:
                annModel.setActivationFunction(ANNModel.STEP);
                break;
            default:
                break;
        }
        if (learningRule == BATCH_GRADIENT_DESCENT || learningRule == DELTA_RULE)
            annModel.setActivationFunction(ANNModel.NO_FUNC);
        annModel.setThreshold(threshold);
        
        // jalankan algoritma
        boolean stop = false;
        iteration = 0;
        
        //annModel.print();
        annModel.resetDeltaWeight();
        do{ 
            if (topology == ONE_PERCEPTRON) {
                switch(learningRule) {
                    case PERCEPTRON_TRAINING_RULE:
                        annModel.perceptronTrainingRule();
                        break;
                    case BATCH_GRADIENT_DESCENT:
                        annModel.batchGradienDescent();
                        break;
                    case DELTA_RULE:
                        annModel.deltaRule();
                        break;
                    default:
                        break;
                }
            } else if (topology == MULTILAYER_PERCEPTRON) {
                annModel.backProp();
            }
            iteration++;
            
            // berhenti jika terminateCondition terpenuhi
            switch(terminationCondition) {
                case TERMINATE_MAX_ITERATION:
                    if(iteration >= maxIteration) stop = true;
                    break;
                case TERMINATE_MSE:
                    if(annModel.error < deltaMSE) stop = true;
                    break;
                case TERMINATE_BOTH:
                    if(iteration > maxIteration || annModel.error < deltaMSE) stop = true;
                    break;
                default:
                    break;
            }
        }while(!stop);
        
//        annModel.print();
    }
    
    /**
     * Fungsi untuk prediksi kelas dari suatu datum setelah model
     * selesai dibangun
     * @param instance datum yang mau diklasifikasi
     * @return array of double berisi nilai dari masing-masing output
     * @throws Exception Exception apapun yang menyebabkan gagal memprediksi
     */
    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        // ubah instance ke Data
        Data data = instanceToData(instance);
        double[] target = new double[data.target.size()];
        if (annModel != null) {
            ArrayList<Double> temp = annModel.calculate(data.input);
            for (int i = 0; i < temp.size(); i++) {
//                System.out.println("temp: "+temp.get(i));
                target[i] = temp.get(i);
            }
        }
        return target;
    }
    
    /**
     * mendapatkan index dari value class jika class adalah nominal
     * @param probability probabilitas dari setiap kelas
     * @return index kelas, jika tidak ada, maka mengembalikan -1
     */
    public int predictClassIndex(double[] probability)  {
        double max = Double.NEGATIVE_INFINITY;
        int idx = -1;
        for (int i = 0; i < probability.length; i++) {
            //debug
//            System.out.println("prob: "+probability[i]);
            if (probability[i] > max) {
                idx = i;
                max = probability[i];
            }
        }

        return idx;
    }
    
    /**
     * mengevaluasi model dengan testSet dan mengembalikan Confusion Matrix
     * buildClassifier harus dipanggil terlebih dahulu
     * @param testSet testSet untuk menguji model
     * @return confusion Matrix, nominal = matrix persegi berukuran NxN dengan
     * N adalah jumlah kelas. numerik = matrix 1x2 dengan elemen pertama adalah 
     * jumlah prediksi yang benar dan elemen kedua adalah jumlah prediksi yang salah
     */
    public int[][] evaluate(Instances testSet) {
        int[][] confusionMatrix;
        if (testSet.classAttribute().isNominal()) {
            confusionMatrix = new int[testSet.classAttribute().numValues()][testSet.classAttribute().numValues()];
        } else {
            confusionMatrix= new int[1][2];
        }
        // debug
        for (int i = 0; i < testSet.numInstances(); i++) {
//            System.out.println("cv: "+testSet.instance(i).classValue());
        }
        
        for (int i = 0; i < testSet.numInstances(); i++) {
            try {
                double[] prob = distributionForInstance(testSet.instance(i));
//                System.out.println("probl:"+prob.length);
//                System.out.println("i: "+testSet.instance(i));
                if (testSet.classAttribute().isNominal()) {
                    int idx = predictClassIndex(prob);
                    confusionMatrix[(int)testSet.instance(i).classValue()][idx]++;
                } else {
                    if (Math.abs(prob[0] - testSet.instance(i).classValue()) <= 0.001)
                        confusionMatrix[0][0]++;
                    else
                        confusionMatrix[0][1]++;
                }
            } catch (Exception ex) {
                Logger.getLogger(MyANN.class.getName()).log(Level.SEVERE, null, ex);
            }
        }
        return confusionMatrix;
    }
    
    /**
     * Mengevaluasi model dengan membagi instances menjadi trainSet dan testSet sebanyak numFold
     * @param instances data yang akan diuji
     * @param numFold
     * @param rand 
     * @return confusion matrix
     */
    public int[][] crossValidation(Instances instances, int numFold, Random rand) {
        int[][] totalResult = null;
        instances = new Instances(instances);
        instances.randomize(rand);
        if (instances.classAttribute().isNominal()) {
            instances.stratify(numFold);
        }
        for (int i = 0; i < numFold; i++) {
            try {
                // membagi instance berdasarkan jumlah fold
                Instances train = instances.trainCV(numFold, i, rand);
                Instances test = instances.testCV(numFold, i);
                MyANN cc = new MyANN(this);
                cc.buildClassifier(train);
                int[][] result = cc.evaluate(test);
                if (i==0) {
                    totalResult = cc.evaluate(test);
                } else {
                    result = cc.evaluate(test);
                    for(int j = 0; j < totalResult.length; j++) {
                        for (int k = 0; k < totalResult[0].length; k++) {
                            totalResult[j][k] += result[j][k];
                        }
                    }
                }
            } catch (Exception ex) {
                Logger.getLogger(MyANN.class.getName()).log(Level.SEVERE, null, ex);
            }
        }
        
        return totalResult;
    }
    
    /**
     * menuliskan error dan iterasi terakhir dari klasifikasi
     */
    public void printSummary() {
        System.out.println("error: "+annModel.error);
        System.out.println("iteration: "+iteration);
    }
    
    /**
     * menuliskan setting dari model
     */
    public void printSetting() {
        System.out.println("activation function: "+paramToString(activationFunction));
        System.out.println("terminate condition: "+paramToString(terminationCondition));
        System.out.println("learning rule: "+paramToString(learningRule));
        System.out.println("topology: "+paramToString(topology));
        System.out.println("delta MSE: "+deltaMSE);
        System.out.println("max iteration: "+maxIteration);
        System.out.println("learning rate: "+learningRate);
        System.out.println("momentum: "+momentum);
        System.out.println("hidden layer: "+(nbLayers.length - 2));
        System.out.print("hidden layer's nodes:");
        for(int i = 1; i < nbLayers.length - 1; i++) {
            System.out.print(nbLayers[i]+" ");
        }
        System.out.println("");
    }
    
    private String paramToString(char param) {
        String retval = "null";
        switch(param) {
            case 1: retval = "one perceptron";
                break;
            case 2: retval = "multi layer perceptron";
                break;
            case 3: retval = "perceptron training rule";
                break;
            case 4: retval = "batch gradient descent";
                break;
            case 5: retval = "delta rule";
                break;
            case 6: retval = "step function";
                break;
            case 7: retval = "sign function";
                break;
            case 8: retval = "sigmoid perceptron";
                break;
            case 9: retval = "terminate using MSE";
                break;
            case 10: retval = "terminate by iteration";
                break;
            case 11: retval = "terminate both way";
                break;
            default:
                break;
        }
        return retval;
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

    /**
     * mengubah instances ke dalam array of data dan disimpan ke variabel datas
     * @param instances input yang akan diubah ke dalam array of data
     */
    private void instancesToDatas(Instances instances) {
        datas = new ArrayList<>();
        
        for (int i = 0; i < instances.numInstances(); i++) {
            datas.add(instanceToData(instances.instance(i)));
        }
    }
    
    /**
     * mengubah Instance menjadi Data
     * @param instance Instance yang akan diubah menjadi kelas Data
     * @return kelas Data dari input
     */
    private Data instanceToData(Instance instance) {
        ArrayList<Double> input = new ArrayList<>();
        ArrayList<Double> target = new ArrayList<>();
        for (int j = 0; j < instance.numAttributes()-1; j++) {
            input.add(0.0);
        }
        if (instance.classAttribute().isNominal()) {
            for (int j = 0; j < instance.classAttribute().numValues(); j++) {
                target.add(0.0);
            }
        } else {
            target.add(0.0);
        }
        for (int j = 0; j < instance.numAttributes(); j++) {
            if (j == instance.classIndex()) {
                if (instance.attribute(j).isNominal())
                    target.set((int)instance.value(j), 1.0);
                else
                    target.add(instance.value(j));
            } else {
                input.set(j, instance.value(j));
            }
        }
        return new Data(input, target);
    }
}
