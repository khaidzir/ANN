/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package ann;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

/**
 *
 * @author khaidzir
 */
public class ANNModel {
    
    /* KODE FUNGSI AKTIVASI */
    public static final int NO_FUNC = 10, STEP = 1, SIGN = 2, SIGMOID = 3 ;
    
    // Atribut - atribut utama model
    private ArrayList<ArrayList<Node>> layers;
    private Map<Node, Map<Node, Double>> weightMap;
    private ArrayList<Double> biases;
    private Map<Integer, Map<Node, Double>> biasesWeight;
    private ArrayList<Data> trainingSet;
    private double learningRate;
    private int activationFuncCode;
    private double threshold;   // threshold buat fungsi step atau sign
    private double momentum;
    
    // Atribut - atribut bantuan
    public double error;
    private double[][] tempTable, errorTable;
    // Map - map buat menyimpan perubahan bobot
    private Map<Node, Map<Node, Double>> mapDWeight;
    private Map<Integer, Map<Node, Double>> mapDBias;
    
    public ANNModel(ArrayList<ArrayList<Node>> layers, Map<Node, Map<Node, Double>> weight, 
                    ArrayList<Double> bias, Map<Integer, Map<Node, Double>> bWeight) {
        this.layers = layers;
        this.weightMap = weight;
        this.biases = bias;
        this.biasesWeight = bWeight;
        
        tempTable = new double[layers.size()-1][];
        errorTable = new double[layers.size()-1][];
        for(int i=1; i<layers.size(); i++) {
            tempTable[i-1] = new double[layers.get(i).size()];
            errorTable[i-1] = new double[layers.get(i).size()];
        }
        activationFuncCode = NO_FUNC;
        momentum = 0.0;
        
        // Map buat nyimpan delta weight kumulatif
        mapDWeight = new HashMap<>();
        for (Node n1 : weightMap.keySet()) {
            Map<Node, Double> map = new HashMap<>();
            for(Node n2 : weightMap.get(n1).keySet()) {                
                map.put(n2, 0.0);
            }
            mapDWeight.put(n1, map);
        }
        
        // Map sementara buat nyimpan delta weight bias kumulatif
        mapDBias = new HashMap<>();        
        for (Integer n1 : biasesWeight.keySet()) {
            Map<Node, Double> map = new HashMap<>();
            for(Node n2 : biasesWeight.get(n1).keySet()) {                
                map.put(n2, 0.0);
            }
            mapDBias.put(n1, map);
        }
    }
    
    // Setter
    public void setDataSet(ArrayList<Data> data) {
        this.trainingSet = data;
    }
    public void setLearningRate(double eta) {
        this.learningRate = eta;
    }
    public void setActivationFunction(int funcCode) {
        this.activationFuncCode = funcCode;
    }
    public void setThreshold(double t) {
        this.threshold = t;
    }
    public void setMomentum(double alpha) {
        this.momentum = alpha;
    }
    
    /* FUNGSI AKTIVASI */
    private double stepFunc(double input) {
        if(input >= threshold) return 1;
        else return 0;
    }
    private double signFunc(double input) {
        if(input >= threshold) return 1;
        else return -1;
    }
    private double sigmoidFunc(double input) {
        return 1.0/(1.0+Math.exp(-input));
    }
    
    /* Menghitung input ke dalam model */
    public ArrayList<Double> calculate(ArrayList<Double> input) {
        feedForward(input);
        ArrayList<Double> output = new ArrayList<>();
        for(int i=0; i<tempTable[layers.size()-2].length; i++) {
            output.add(tempTable[layers.size()-2][i]);
        }
        return output;
    }
    
    /* Feed forward untuk learning */
    private void feedForward(ArrayList<Double> input) {
        // Input pertama
        for(int i=0; i<layers.get(1).size(); i++) {
            tempTable[0][i] = 0.0;
            for(int j=0; j<layers.get(0).size(); j++) {
//                System.out.println(input.get(j)+" * "+ weightMap.get(layers.get(0).get(j)).get(layers.get(1).get(i)));
                tempTable[0][i] += input.get(j) * weightMap.get(layers.get(0).get(j)).get(layers.get(1).get(i));
            }
//            System.out.println(biases.get(0)+" * "+ biasesWeight.get(0).get(layers.get(1).get(i)));
            tempTable[0][i] += biases.get(0)*biasesWeight.get(0).get(layers.get(1).get(i));
//            System.out.println("Hasil 1 : " + tempTable[0][i]);
            
            if(activationFuncCode == STEP) tempTable[0][i] = stepFunc(tempTable[0][i]);
            else if(activationFuncCode == SIGN) tempTable[0][i] = signFunc(tempTable[0][i]);
            else if (activationFuncCode == SIGMOID) tempTable[0][i] = sigmoidFunc(tempTable[0][i]);
//            System.out.println("Hasil 2 : " + tempTable[0][i]);
//            System.out.println("----");            
        }
        
        for(int k=1; k<layers.size()-1; k++) {
            for(int i=0; i<layers.get(k+1).size(); i++) {
                tempTable[k][i] = 0.0;
                for(int j=0; j<layers.get(k).size(); j++) {   
//                    System.out.println(tempTable[k-1][j]+" * "+ weightMap.get(layers.get(k).get(j)).get(layers.get(k+1).get(i)));
                    tempTable[k][i] += tempTable[k-1][j] * weightMap.get(layers.get(k).get(j)).get(layers.get(k+1).get(i));
                }
//                System.out.println(biases.get(k)+" * "+ biasesWeight.get(k).get(layers.get(k+1).get(i)));
                tempTable[k][i] += biases.get(k)*biasesWeight.get(k).get(layers.get(k+1).get(i));
//                System.out.println("Hasil 1 : " + tempTable[k][i]);

                if(activationFuncCode == STEP) tempTable[k][i] = stepFunc(tempTable[k][i]);
                else if(activationFuncCode == SIGN) tempTable[k][i] = signFunc(tempTable[k][i]);
                else if (activationFuncCode == SIGMOID) tempTable[k][i] = sigmoidFunc(tempTable[k][i]);
                
//                System.out.println("Hasil 2 : " + tempTable[k][i]);
//                System.out.println("----");
            }
        }
    }
    
    /* Reset semua delta weight yang disimpan */
    public void resetDeltaWeight() {
        // Reset map delta weight
        for(int i=0; i<layers.size()-1; i++) {
            for(int j=0; j<layers.get(i).size(); j++) {
                for(int k=0; k<layers.get(i+1).size(); k++) {
                    mapDWeight.get(layers.get(i).get(j)).replace(layers.get(i+1).get(k), 0.0);
                }
            }
        }
        
        // Reset map delta bias
        for(int i=0; i<layers.size()-1; i++) {
            for(int j=0; j<layers.get(i+1).size(); j++) {
                mapDBias.get(i).replace(layers.get(i+1).get(j), 0.0);
            }
        }
    }
    
    /* MULTI LAYER PERCEPTRON */
    /* Perhatian! Pastikan mapDWeight & mapDBias semua bernilai 0 sebelum memanggil ini ! 
       (Untuk memastikan, tinggal panggil resetDeltaWeight()) */
    public void backProp() {
        double output, dW, weight;
        error = 0.0;
        
        // Semua data set
        for(Data d : trainingSet) {
            feedForward(d.input);
            
            for(int i=0; i<tempTable[tempTable.length-1].length; i++) {
                output = (d.target.get(i)-tempTable[tempTable.length-1][i]);
                error += output * output;
            }
            
            // Hitung error output
            for(int i=0; i<layers.get(layers.size()-1).size(); i++) {
                output = tempTable[tempTable.length-1][i];
                errorTable[errorTable.length-1][i] = output * (1-output) * (d.target.get(i)-output);
            }
            
            // Hitung error hidden layer
            for(int i=layers.size()-2; i>0; i--) {
                for(int j=0; j<layers.get(i).size(); j++) {
                    output = tempTable[i-1][j];
                    errorTable[i-1][j] = output * (1-output);
                    output = 0.0;
                    for(int k=0; k<layers.get(i+1).size(); k++) {                        
                        output += errorTable[i][k] * 
                                weightMap.get(layers.get(i).get(j)).get(layers.get(i+1).get(k));
                    }
                    errorTable[i-1][j] *= output;
                }
            }
            
            // Update weight input            
            for(int i=0; i<layers.get(0).size(); i++) {
                for(int j=0; j<layers.get(1).size(); j++) {
//                    System.out.println("PrevDWeight = " + mapDWeight.get(layers.get(0).get(i)).get(layers.get(1).get(j)));
                    dW = ( learningRate * errorTable[0][j] * d.input.get(i) ) + 
                         ( momentum * mapDWeight.get(layers.get(0).get(i)).get(layers.get(1).get(j)) );
                    weight = weightMap.get(layers.get(0).get(i)).get(layers.get(1).get(j));
                    weightMap.get(layers.get(0).get(i)).replace(layers.get(1).get(j), weight+dW);
                    mapDWeight.get(layers.get(0).get(i)).replace(layers.get(1).get(j), dW);
//                    System.out.println("DW : " + dW);
                }
            }
            
            // debug
            /*
            System.out.println("tempT: "+tempTable.length);
            for(double[] dtd : tempTable) {
                for (double ddt : dtd) {
                    System.out.print(ddt+" ");
                }
                System.out.println("");
            }
            System.out.println("\n");
            System.out.println("errT: "+errorTable.length);
            for(double[] dtd : errorTable) {
                for (double ddt : dtd) {
                    System.out.print(ddt+" ");
                }
                System.out.println("");
            }
            System.out.println("\n");
            System.out.println("layers: "+layers.size());
            for(ArrayList<Node> dtd : layers) {
                for (Node ddt : dtd) {
                    System.out.print(ddt.id+" ");
                }
                System.out.println("");
            }
            System.out.println("\n");
            */
            // Update weight sisanya
            for(int k=1; k<layers.size()-1; k++) {
                for(int i=0; i<layers.get(k).size(); i++) {
                    for(int j=0; j<layers.get(k+1).size(); j++) {
//                        System.out.println("PrevDWeight = " + mapDWeight.get(layers.get(k).get(i)).get(layers.get(k+1).get(j)));
                        // debug
                       // System.out.println("k: "+k+", i: "+i+", j: "+j);
                                
                        dW = ( learningRate * errorTable[k][j] * tempTable[k-1][i] ) +  
                             ( momentum * mapDWeight.get(layers.get(k).get(i)).get(layers.get(k+1).get(j)) );
                        weight = weightMap.get(layers.get(k).get(i)).get(layers.get(k+1).get(j));
                        weightMap.get(layers.get(k).get(i)).replace(layers.get(k+1).get(j), weight+dW);
                        mapDWeight.get(layers.get(k).get(i)).replace(layers.get(k+1).get(j), dW);
//                        System.out.println("DW : " + dW);
                    }
                }
            }
            
            // Update weight bias
            for(int k=0; k<layers.size()-1; k++) {
                for(int j=0; j<layers.get(k+1).size(); j++) {
                    dW = ( learningRate * errorTable[k][j] * biases.get(k) ) + 
                         ( momentum * mapDBias.get(k).get(layers.get(k+1).get(j)) );
                    output = biasesWeight.get(k).get(layers.get(k+1).get(j));
                    biasesWeight.get(k).replace(layers.get(k+1).get(j), output+dW);
                    mapDBias.get(k).replace(layers.get(k+1).get(j), dW);
                }
            }
        }
        error /= 2;
//        System.out.println("Error sekarang : " + error);
    }
    
    /* SINGLE LAYER PERCEPTRON */
    public void perceptronTrainingRule() {
        double output, dW;
        error = 0.0;
        for(Data d : trainingSet) {
            feedForward(d.input);
            
            for(int i=0; i<tempTable[0].length; i++) {
                output = (d.target.get(i)-tempTable[tempTable.length-1][i]);
                error += output * output;
            }
            
            // Hitung error output
            for(int i=0; i<layers.get(1).size(); i++) {
                output = tempTable[tempTable.length-1][i];
                errorTable[errorTable.length-1][i] = (d.target.get(i)-output);
                if(activationFuncCode == SIGMOID)
                    errorTable[errorTable.length-1][i] *= output * (1-output);
            }
            
            // Update weight input            
            for(int i=0; i<layers.get(0).size(); i++) {
                for(int j=0; j<layers.get(1).size(); j++) {
                    dW = learningRate * errorTable[0][j] * d.input.get(i);
                    output = weightMap.get(layers.get(0).get(i)).get(layers.get(1).get(j));
                    weightMap.get(layers.get(0).get(i)).replace(layers.get(1).get(j), output+dW);
                }
            }
            
            // Update weight bias
            for(int k=0; k<layers.size()-1; k++) {
                for(int j=0; j<layers.get(k+1).size(); j++) {
                    dW = learningRate * errorTable[k][j] * biases.get(k);
                    output = biasesWeight.get(k).get(layers.get(k+1).get(j));
                    biasesWeight.get(k).replace(layers.get(k+1).get(j), output+dW);
                }
            }
        }
        error /= 2;
//        System.out.println("Error sekarang : " + error);
    }
    
    public void batchGradienDescent() {
        double output, dW;
        error = 0.0;
        
        // Reset map delta weight        
        for(int i=0; i<layers.get(0).size(); i++) {
            for(int j=0; j<layers.get(1).size(); j++) {
                mapDWeight.get(layers.get(0).get(i)).replace(layers.get(1).get(j), 0.0);
            }
        }
        
        // Reset map delta bias
        for(int i=0; i<layers.get(1).size(); i++) {
            mapDBias.get(0).replace(layers.get(1).get(i), 0.0);
        }
        
        // Semua data set
        for(Data d : trainingSet) {
            feedForward(d.input);
            
            // Hitung error
            for(int i=0; i<tempTable[tempTable.length-1].length; i++) {
                output = (d.target.get(i)-tempTable[tempTable.length-1][i]);
//                System.out.println("output: "+output);
                error += output * output;
            }
            
            // Hitung delta weight
            for(int i=0; i<layers.get(0).size(); i++) {
                for(int j=0; j<layers.get(1).size(); j++) {
                    output = tempTable[0][j];
                    dW = learningRate * (d.target.get(j)-output) * d.input.get(i);
                    if(activationFuncCode == SIGMOID)
                        dW *= output * (1-output);
                    output = mapDWeight.get(layers.get(0).get(i)).get(layers.get(1).get(j));
                    mapDWeight.get(layers.get(0).get(i)).replace(layers.get(1).get(j), output+dW);
                }
            }
            
            // Hitung delta weight bias
            for(int j=0; j<layers.get(1).size(); j++) {
                output = tempTable[0][j];
                dW = learningRate * (d.target.get(j)-output) * biases.get(0);
                if(activationFuncCode == SIGMOID)
                    dW *= output * (1-output);
                output = mapDBias.get(0).get(layers.get(1).get(j));
                mapDBias.get(0).replace(layers.get(1).get(j), output+dW);
            }
        }
            
        // Update weight input            
        for(int i=0; i<layers.get(0).size(); i++) {
            for(int j=0; j<layers.get(1).size(); j++) {
                dW = mapDWeight.get(layers.get(0).get(i)).get(layers.get(1).get(j));
                output = weightMap.get(layers.get(0).get(i)).get(layers.get(1).get(j));
                weightMap.get(layers.get(0).get(i)).replace(layers.get(1).get(j), output+dW);
            }
        }

        // Update weight bias
        for(int j=0; j<layers.get(1).size(); j++) {
            dW = mapDBias.get(0).get(layers.get(1).get(j));
            output = biasesWeight.get(0).get(layers.get(1).get(j));
            biasesWeight.get(0).replace(layers.get(1).get(j), output+dW);
        }
        
        error /= 2;
//        System.out.println("Error sekarang : " + error);
    }
    
    public void deltaRule() {
        double output, dW;
        error = 0.0;
        for(Data d : trainingSet) {
            feedForward(d.input);
            
            for(int i=0; i<tempTable[0].length; i++) {
                output = (d.target.get(i)-tempTable[tempTable.length-1][i]);
//                System.out.println("output: "+output);
                error += output * output;
            }
            
            // Hitung error output
            for(int i=0; i<layers.get(1).size(); i++) {
                output = tempTable[tempTable.length-1][i];
                errorTable[errorTable.length-1][i] = (d.target.get(i)-output);
                if(activationFuncCode == SIGMOID)
                    errorTable[errorTable.length-1][i] *= output * (1-output);
            }
            
            // Update weight input            
            for(int i=0; i<layers.get(0).size(); i++) {
                for(int j=0; j<layers.get(1).size(); j++) {
                    dW = learningRate * errorTable[0][j] * d.input.get(i);
                    output = weightMap.get(layers.get(0).get(i)).get(layers.get(1).get(j));
                    weightMap.get(layers.get(0).get(i)).replace(layers.get(1).get(j), output+dW);
                }
            }
            
            // Update weight bias
            for(int k=0; k<layers.size()-1; k++) {
                for(int j=0; j<layers.get(k+1).size(); j++) {
                    dW = learningRate * errorTable[k][j] * biases.get(k);
                    output = biasesWeight.get(k).get(layers.get(k+1).get(j));
                    biasesWeight.get(k).replace(layers.get(k+1).get(j), output+dW);
                }
            }
        }
        error /= 2;
//        System.out.println("Error sekarang : " + error);
    }
    
    
    public void print() {
        //////////////////////////////////////////////////////////////////////
        System.out.println("Node : ");
        for(Map.Entry<Node, Map<Node, Double>> me : weightMap.entrySet()) {
            Map<Node, Double> map = me.getValue();
            for(Map.Entry<Node, Double> en : map.entrySet()) {
                System.out.println(me.getKey().getID() + " - " + en.getKey().getID() + " : " + en.getValue());
            }
        }
        
        System.out.println("Bias : ");
        for(Map.Entry<Integer, Map<Node, Double>> me : biasesWeight.entrySet()) {
            Map<Node, Double> map = me.getValue();
            for(Map.Entry<Node, Double> en : map.entrySet()) {
                System.out.println(me.getKey() + " - " + en.getKey().getID() + " : " + en.getValue());
            }
        }
        //////////////////////////////////////////////////////////////////////
    }
    
    
    public static void main(String[] args) {
        coba4();
    }
    
    public static void coba() {
        String id = "node";
        
        int nlayer=2;
        ArrayList<Integer> nNode = new ArrayList<>();
        nNode.add(2);   // 2 input
        nNode.add(2);   // 2 node layer awal
        nNode.add(2);   // 2 node layer akhir
        
        /* Buat layer */
        ArrayList<ArrayList<Node>> layers = new ArrayList<>();
        for(int i=0; i<=nlayer; i++) {
            layers.add(new ArrayList<>());
        }
        
        // Inisialisasi bagian input
        for(int i=0; i<nNode.get(0); i++)
            layers.get(0).add(new Node(id+"-"+0+"-"+i, null, layers.get(1)));
        
        // Inisialisasi layer awal-tengah
        int i=1;
        for(; i<nNode.size()-1; i++) {
            for(int j=0; j<nNode.get(i); j++) {
                layers.get(i).add(new Node(id+"-"+i+"-"+j, layers.get(i-1), layers.get(i+1)));
            }
        }
        
        // Inisialisasi layer akhir
        for(int j=0; j<nNode.get(i); j++) {
            layers.get(i).add(new Node(id+"-"+i+"-"+j, layers.get(i-1), null));
        }
        
        /* Buat weight */
        
        // weight tiap hubungan neuron
        ArrayList<Double> listWeight = new ArrayList<>();
        listWeight.add(0.15);
        listWeight.add(0.25);
        listWeight.add(0.20);
        listWeight.add(0.30);
        
        listWeight.add(0.40);
        listWeight.add(0.50);
        listWeight.add(0.45);
        listWeight.add(0.55);
        
        // nilai bias
        ArrayList<Double> bias = new ArrayList<>();
        bias.add(1.0);
        bias.add(1.0);
        
        // daftar bobot bias
        ArrayList<Double> listWeightBias = new ArrayList<>();
        listWeightBias.add(0.35);
        listWeightBias.add(0.35);
        listWeightBias.add(0.60);
        listWeightBias.add(0.60);
        
        // Bobot bias
        Map<Integer, Map<Node, Double>> biasesWeight = new HashMap<>();
        i=0;
        for(int j=0; j<layers.size()-1; j++) {
            ArrayList<Node> arrNode = layers.get(j+1);
            Map<Node, Double> map = new HashMap<>();
            for(Node node : arrNode) {                
                map.put(node, listWeightBias.get(i));
                i++;
            }
            biasesWeight.put(j, map);            
        }
        
        // map weight
        Map<Node, Map<Node, Double>> mapWeight = new HashMap<>();
        
        // Isi tiap weight
        i=0;        
        for(int j=0; j<layers.size()-1; j++) {
            ArrayList<Node> arrNode = layers.get(j);
            for(Node node : arrNode) {
                Map<Node, Double> map = new HashMap<>();
                for(Node nextNode : node.getNextNodes()) {                    
                    map.put(nextNode, listWeight.get(i));
                    i++;
                }
                mapWeight.put(node, map);
            }
        }
        
        ANNModel annModel = new ANNModel(layers, mapWeight, bias, biasesWeight);
        annModel.setActivationFunction(ANNModel.SIGMOID);
        ArrayList<Double> input = new ArrayList<>();
        input.add(0.05);
        input.add(0.10);
        ArrayList<Double> arr = annModel.calculate(input);
        for(Double d : arr) {
            System.out.println(d);
        }
        annModel.print();
    }
    
    public static void coba2() {
        String id = "node";
        
        int nlayer=2;
        ArrayList<Integer> nNode = new ArrayList<>();
        nNode.add(2);   // 2 input
        nNode.add(2);   // 2 node layer awal
        nNode.add(2);   // 2 node layer akhir
        
        /* Buat layer */
        ArrayList<ArrayList<Node>> layers = new ArrayList<>();
        for(int i=0; i<=nlayer; i++) {
            layers.add(new ArrayList<>());
        }
        
        // Inisialisasi bagian input
        for(int i=0; i<nNode.get(0); i++)
            layers.get(0).add(new Node(id+"-"+0+"-"+i, null, layers.get(1)));
        
        // Inisialisasi layer awal-tengah
        int i=1;
        for(; i<nNode.size()-1; i++) {
            for(int j=0; j<nNode.get(i); j++) {
                layers.get(i).add(new Node(id+"-"+i+"-"+j, layers.get(i-1), layers.get(i+1)));
            }
        }
        
        // Inisialisasi layer akhir
        for(int j=0; j<nNode.get(i); j++) {
            layers.get(i).add(new Node(id+"-"+i+"-"+j, layers.get(i-1), null));
        }
        
        /* Buat weight */
        
        // weight tiap hubungan neuron
        ArrayList<Double> listWeight = new ArrayList<>();
        listWeight.add(0.15);
        listWeight.add(0.25);
        listWeight.add(0.20);
        listWeight.add(0.30);
        
        listWeight.add(0.40);
        listWeight.add(0.50);
        listWeight.add(0.45);
        listWeight.add(0.55);
        
        // nilai bias
        ArrayList<Double> bias = new ArrayList<>();
        bias.add(1.0);
        bias.add(1.0);
        
        // daftar bobot bias
        ArrayList<Double> listWeightBias = new ArrayList<>();
        listWeightBias.add(0.35);
        listWeightBias.add(0.35);
        listWeightBias.add(0.60);
        listWeightBias.add(0.60);
        
        // Bobot bias
        Map<Integer, Map<Node, Double>> biasesWeight = new HashMap<>();
        i=0;
        for(int j=0; j<layers.size()-1; j++) {
            ArrayList<Node> arrNode = layers.get(j+1);
            Map<Node, Double> map = new HashMap<>();
            for(Node node : arrNode) {                
                map.put(node, listWeightBias.get(i));
                i++;
            }
            biasesWeight.put(j, map);            
        }
        
        // map weight
        Map<Node, Map<Node, Double>> mapWeight = new HashMap<>();
        
        // Isi tiap weight
        i=0;        
        for(int j=0; j<layers.size()-1; j++) {
            ArrayList<Node> arrNode = layers.get(j);
            for(Node node : arrNode) {
                Map<Node, Double> map = new HashMap<>();
                for(Node nextNode : node.getNextNodes()) {                    
                    map.put(nextNode, listWeight.get(i));
                    i++;
                }
                mapWeight.put(node, map);
            }
        }
        
        ANNModel annModel = new ANNModel(layers, mapWeight, bias, biasesWeight);
        ArrayList<Double> input = new ArrayList<>();
        input.add(0.01);
        input.add(0.99);
        
        ArrayList<Double> output = new ArrayList<>();
        output.add(0.0);
        output.add(1.0);
        
        ArrayList<Data> trainingSet = new ArrayList<>();
        Data d = new Data();
        d.input = input;
        d.target = output;
        trainingSet.add(d);
        
        annModel.setDataSet(trainingSet);
        annModel.setLearningRate(0.1);
        annModel.setActivationFunction(ANNModel.SIGMOID);
        annModel.setMomentum(0.5);
        annModel.setThreshold(0.0);
        System.out.println("Awal : ");
        annModel.print();
        
        System.out.println("--\nBackProp : ");
        int counter = 0;
        do {
            annModel.backProp();
//            annModel.print();
            counter++;
        } while(annModel.error > 0.001);
        System.out.println("\n--Jumlah Iterasi : " + counter);
        
    }
    
    // Single layer perceptron
    public static void coba3() {
        String id = "node";
        
        int nlayer=1;
        ArrayList<Integer> nNode = new ArrayList<>();
        nNode.add(3);   // 3 input
        nNode.add(1);   // 1 node layer awal
        
        /* Buat layer */
        ArrayList<ArrayList<Node>> layers = new ArrayList<>();
        for(int i=0; i<=nlayer; i++) {
            layers.add(new ArrayList<>());
        }
        
        // Inisialisasi bagian input
        for(int i=0; i<nNode.get(0); i++)
            layers.get(0).add(new Node(id+"-"+0+"-"+i, null, layers.get(1)));
        
        // Inisialisasi layer awal-tengah
        int i=1;
        for(; i<nNode.size()-1; i++) {
            for(int j=0; j<nNode.get(i); j++) {
                layers.get(i).add(new Node(id+"-"+i+"-"+j, layers.get(i-1), layers.get(i+1)));
            }
        }
        
        // Inisialisasi layer akhir
        for(int j=0; j<nNode.get(i); j++) {
            layers.get(i).add(new Node(id+"-"+i+"-"+j, layers.get(i-1), null));
        }
        
        /* Buat weight */
        
        // weight tiap hubungan neuron
        ArrayList<Double> listWeight = new ArrayList<>();
        listWeight.add(0.0);
        listWeight.add(0.0);
        listWeight.add(0.0);
        
        // nilai bias
        ArrayList<Double> bias = new ArrayList<>();
        bias.add(1.0);
        
        // daftar bobot bias
        ArrayList<Double> listWeightBias = new ArrayList<>();
        listWeightBias.add(0.0);    
        
        // Bobot bias
        Map<Integer, Map<Node, Double>> biasesWeight = new HashMap<>();
        i=0;
        for(int j=0; j<layers.size()-1; j++) {
            ArrayList<Node> arrNode = layers.get(j+1);
            Map<Node, Double> map = new HashMap<>();
            for(Node node : arrNode) {                
                map.put(node, listWeightBias.get(i));
                i++;
            }
            biasesWeight.put(j, map);            
        }
        
        // map weight
        Map<Node, Map<Node, Double>> mapWeight = new HashMap<>();
        
        // Isi tiap weight
        i=0;        
        for(int j=0; j<layers.size()-1; j++) {
            ArrayList<Node> arrNode = layers.get(j);
            for(Node node : arrNode) {
                Map<Node, Double> map = new HashMap<>();
                for(Node nextNode : node.getNextNodes()) {                    
                    map.put(nextNode, listWeight.get(i));
                    i++;
                }
                mapWeight.put(node, map);
            }
        }
        
        ANNModel annModel = new ANNModel(layers, mapWeight, bias, biasesWeight);
        
        /* Set training set */
        ArrayList<Data> trainingSet = new ArrayList<>();
        
        /* Dataset ke-1 */
        ArrayList<Double> input1 = new ArrayList<>();
        input1.add(1.0);
        input1.add(0.0);
        input1.add(1.0);
        
        ArrayList<Double> output1 = new ArrayList<>();
        output1.add(-1.0);
        
        Data d1 = new Data();
        d1.input = input1;
        d1.target = output1;
        trainingSet.add(d1);
        
        /* Dataset ke-2 */
        ArrayList<Double> input2 = new ArrayList<>();
        input2.add(0.0);
        input2.add(-1.0);
        input2.add(-1.0);
        
        ArrayList<Double> output2 = new ArrayList<>();
        output2.add(1.0);
        
        Data d2 = new Data();
        d2.input = input2;
        d2.target = output2;
        trainingSet.add(d2);
        
        /* Dataset ke-3 */
        ArrayList<Double> input3 = new ArrayList<>();
        input3.add(-1.0);
        input3.add(-0.5);
        input3.add(-1.0);
        
        ArrayList<Double> output3 = new ArrayList<>();
        output3.add(1.0);
        
        Data d3 = new Data();
        d3.input = input3;
        d3.target = output3;
        trainingSet.add(d3);
        
        annModel.setDataSet(trainingSet);
        annModel.setLearningRate(0.1);
        annModel.setActivationFunction(ANNModel.SIGN);
        annModel.setThreshold(0.0);
        System.out.println("Awal : ");
//        annModel.print();
        
        System.out.println("--\nBackProp : ");
        int counter = 0;
        do {
            annModel.perceptronTrainingRule();
            annModel.print();
            counter++;
        } while(annModel.error > 0.01);
        System.out.println("\n--Jumlah Iterasi : " + counter);

//        ArrayList<Double> arr = annModel.calculate(input);
//        for(Double a : arr) {
//            System.out.println(a);
//        }
//        annModel.print();
        
    }
 
    public static void coba4() {
        String id = "node";
        
        int nlayer=2;
        ArrayList<Integer> nNode = new ArrayList<>();
        nNode.add(2);   // 2 input
        nNode.add(3);   // 2 node layer awal
        nNode.add(3);   // 2 node layer akhir
        
        /* Buat layer */
        ArrayList<ArrayList<Node>> layers = new ArrayList<>();
        for(int i=0; i<=nlayer; i++) {
            layers.add(new ArrayList<>());
        }
        
        // Inisialisasi bagian input
        for(int i=0; i<nNode.get(0); i++)
            layers.get(0).add(new Node(id+"-"+0+"-"+i, null, layers.get(1)));
        
        // Inisialisasi layer awal-tengah
        int i=1;
        for(; i<nNode.size()-1; i++) {
            for(int j=0; j<nNode.get(i); j++) {
                layers.get(i).add(new Node(id+"-"+i+"-"+j, layers.get(i-1), layers.get(i+1)));
            }
        }
        
        // Inisialisasi layer akhir
        for(int j=0; j<nNode.get(i); j++) {
            layers.get(i).add(new Node(id+"-"+i+"-"+j, layers.get(i-1), null));
        }
        
        /* Buat weight */
        
        // weight tiap hubungan neuron
        ArrayList<Double> listWeight = new ArrayList<>();
        listWeight.add(0.15);
        listWeight.add(0.25);
        listWeight.add(0.10);
        listWeight.add(0.20);
        listWeight.add(0.30);
        listWeight.add(0.15);
        
        listWeight.add(0.40);
        listWeight.add(0.50);
        listWeight.add(0.45);
        listWeight.add(0.55);
        listWeight.add(0.40);
        listWeight.add(0.50);
        
                listWeight.add(0.40);
        listWeight.add(0.50);
        listWeight.add(0.45);
        listWeight.add(0.55);
        listWeight.add(0.40);
        listWeight.add(0.50);
        
        // nilai bias
        ArrayList<Double> bias = new ArrayList<>();
        bias.add(1.0);
        bias.add(1.0);
        
        // daftar bobot bias
        ArrayList<Double> listWeightBias = new ArrayList<>();
        listWeightBias.add(0.35);
        listWeightBias.add(0.35);
        listWeightBias.add(0.60);
        listWeightBias.add(0.60);
        listWeightBias.add(0.60);
        
        listWeightBias.add(0.35);
        listWeightBias.add(0.35);
        listWeightBias.add(0.60);
        listWeightBias.add(0.60);
        listWeightBias.add(0.60);
        
        // Bobot bias
        Map<Integer, Map<Node, Double>> biasesWeight = new HashMap<>();
        i=0;
        for(int j=0; j<layers.size()-1; j++) {
            ArrayList<Node> arrNode = layers.get(j+1);
            Map<Node, Double> map = new HashMap<>();
            for(Node node : arrNode) {                
                map.put(node, listWeightBias.get(i));
                i++;
            }
            biasesWeight.put(j, map);            
        }
        
        // map weight
        Map<Node, Map<Node, Double>> mapWeight = new HashMap<>();
        
        // Isi tiap weight
        i=0;        
        for(int j=0; j<layers.size()-1; j++) {
            ArrayList<Node> arrNode = layers.get(j);
            for(Node node : arrNode) {
                Map<Node, Double> map = new HashMap<>();
                for(Node nextNode : node.getNextNodes()) {                    
                    map.put(nextNode, listWeight.get(i));
                    i++;
                }
                mapWeight.put(node, map);
            }
        }
        
        ANNModel annModel = new ANNModel(layers, mapWeight, bias, biasesWeight);
        ArrayList<Double> input = new ArrayList<>();
        input.add(0.01);
        input.add(0.99);
        
        ArrayList<Double> output = new ArrayList<>();
        output.add(0.0);
        output.add(1.0);
        output.add(1.0);
        
        ArrayList<Data> trainingSet = new ArrayList<>();
        Data d = new Data();
        d.input = input;
        d.target = output;
        trainingSet.add(d);
        
        annModel.setDataSet(trainingSet);
        annModel.setLearningRate(0.1);
        annModel.setActivationFunction(ANNModel.SIGMOID);
        annModel.setMomentum(0.5);
        annModel.setThreshold(0.0);
        System.out.println("Awal : ");
        annModel.print();
        
        System.out.println("--\nBackProp : ");
        int counter = 0;
        do {
            annModel.backProp();
//            annModel.print();
            counter++;
        } while(annModel.error > 0.001);
        System.out.println("\n--Jumlah Iterasi : " + counter);
    }
    
}