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
    
    private ArrayList<ArrayList<Node>> layers;
    private Map<Node, Map<Node, Double>> weightMap;
    private ArrayList<Double> biases;
    private ArrayList<Double> biasesWeight;
    
    private double[][] tempTable, errorTable;
    private double learningRate;
    public double error;
    
    private ArrayList<Data> trainingSet;
    
    public ANNModel(ArrayList<ArrayList<Node>> layers, Map<Node, Map<Node, Double>> weight, 
                    ArrayList<Double> bias, ArrayList<Double> bWeight) {
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
    }
    
    // Setter
    public void setDataSet(ArrayList<Data> data) {
        this.trainingSet = data;
    }
    public void setLearningRate(double eta) {
        learningRate = eta;
    }
    
    public ArrayList<Double> calculate(ArrayList<Double> input) {
        feedForward(input);
        ArrayList<Double> output = new ArrayList<>();
        for(int i=0; i<tempTable[layers.size()-2].length; i++) {
            output.add(tempTable[layers.size()-2][i]);
        }
        return output;
    }
    
    private void feedForward(ArrayList<Double> input) {
        // Input pertama
        int b = 0;
        for(int i=0; i<layers.get(1).size(); i++) {
            tempTable[0][i] = 0.0;
            for(int j=0; j<layers.get(0).size(); j++) {
//                System.out.println(input.get(j)+" * "+ weightMap.get(layers.get(0).get(j)).get(layers.get(1).get(i)));
                tempTable[0][i] += input.get(j) * weightMap.get(layers.get(0).get(j)).get(layers.get(1).get(i));
            }
//            System.out.println(biases.get(0)+" * "+ biasesWeight.get(b));
            tempTable[0][i] += biases.get(0)*biasesWeight.get(b);
//            System.out.println("Hasil 1 : " + tempTable[0][i]);
            b++;
            tempTable[0][i] = sigmoidFunc(tempTable[0][i]);
//            System.out.println("Hasil 2 : " + tempTable[0][i]);
//            System.out.println("----");            
        }
        
        for(int k=1; k<layers.size()-1; k++) {
            for(int i=0; i<layers.get(1).size(); i++) {
                tempTable[k][i] = 0.0;
                for(int j=0; j<layers.get(k).size(); j++) {   
//                    System.out.println(tempTable[k-1][j]+" * "+ weightMap.get(layers.get(k).get(j)).get(layers.get(k+1).get(i)));
                    tempTable[k][i] += tempTable[k-1][j] * weightMap.get(layers.get(k).get(j)).get(layers.get(k+1).get(i));
                }
//                System.out.println(biases.get(k)+" * "+ biasesWeight.get(b));
                tempTable[k][i] += biases.get(k)*biasesWeight.get(b);
//                System.out.println("Hasil 1 : " + tempTable[k][i]);
                b++;
                tempTable[k][i] = sigmoidFunc(tempTable[k][i]);
//                System.out.println("Hasil 2 : " + tempTable[k][i]);
//                System.out.println("----");
            }
        }
    }
    
    public void backProp() {
        double output, dW;
        error = 0.0;
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
                    dW = learningRate * errorTable[0][j] * d.input.get(i);
                    output = weightMap.get(layers.get(0).get(i)).get(layers.get(1).get(j));
                    weightMap.get(layers.get(0).get(i)).replace(layers.get(1).get(j), output+dW);
                }
            }
            
            // Update weight sisanya
            for(int k=1; k<layers.size()-1; k++) {
                for(int i=0; i<layers.get(k).size(); i++) {
                    for(int j=0; j<layers.get(k+1).size(); j++) {
                        dW = learningRate * errorTable[k][j] * tempTable[k-1][j];
                        output = weightMap.get(layers.get(k).get(i)).get(layers.get(k+1).get(j));
                        weightMap.get(layers.get(k).get(i)).replace(layers.get(k+1).get(j), output+dW);
                    }
                }
            }
        }
        error /= 2;
        System.out.println("Error sekarang : " + error);
    }
    
    public void print() {
        //////////////////////////////////////////////////////////////////////
        for(Map.Entry<Node, Map<Node, Double>> me : weightMap.entrySet()) {
            Map<Node, Double> map = me.getValue();
            for(Map.Entry<Node, Double> en : map.entrySet()) {
                System.out.println(me.getKey().getID() + " - " + en.getKey().getID() + " : " + en.getValue());
            }
        }
        for(double d : biasesWeight) {
            System.out.println(d);
        }
        //////////////////////////////////////////////////////////////////////
    }
    
    private double sigmoidFunc(double input) {
        return 1.0/(1.0+Math.exp(-input));
    }
    
    public static void main(String[] args) {
        coba2();
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
        
        // bobot bias
        ArrayList<Double> listWeightBias = new ArrayList<>();
        listWeightBias.add(0.35);
        listWeightBias.add(0.35);
        listWeightBias.add(0.60);
        listWeightBias.add(0.60);
        
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
        
        ANNModel annModel = new ANNModel(layers, mapWeight, bias, listWeightBias);
        ArrayList<Double> input = new ArrayList<>();
        input.add(0.05);
        input.add(0.10);
        ArrayList<Double> arr = annModel.calculate(input);
        for(Double d : arr) {
            System.out.println(d);
        }
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
        
        // bobot bias
        ArrayList<Double> listWeightBias = new ArrayList<>();
        listWeightBias.add(0.35);
        listWeightBias.add(0.35);
        listWeightBias.add(0.60);
        listWeightBias.add(0.60);
        
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
        
        ANNModel annModel = new ANNModel(layers, mapWeight, bias, listWeightBias);
        ArrayList<Double> input = new ArrayList<>();
        input.add(0.05);
        input.add(0.10);
        
        ArrayList<Double> output = new ArrayList<>();
        output.add(0.1);
        output.add(0.5);
        
        ArrayList<Data> trainingSet = new ArrayList<>();
        Data d = new Data();
        d.input = input;
        d.target = output;
        trainingSet.add(d);
        
        annModel.setDataSet(trainingSet);
        annModel.setLearningRate(0.1);
        System.out.println("Awal : ");
        annModel.print();
        
        System.out.println("--\nBackProp : ");
        int counter = 1;
        do {
            annModel.backProp();
            annModel.print();
            counter++;
        } while(annModel.error > 0.001);
        System.out.println("\n--Jumlah Iterasi : " + counter);
        
    }
    
}