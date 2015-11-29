/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package ann;

import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 *
 * @author yusuf
 */
public class WeightParser {
    
    public double[][] weight;
    
    WeightParser(int mainWeight, int biasWeight) {
        weight = new double[2][];
        weight[0] = new double[mainWeight];
        weight[1] = new double[biasWeight];
    }
    
    WeightParser(String filename) {
        try {
            List<String> lines = Files.readAllLines(Paths.get(filename), Charset.defaultCharset());
            weight = new double[2][];
            int i = 0;
            for (String line : lines) {
                String[] ws = line.split(" ");
                weight[i] = new double[ws.length];
                for (int j = 0; j < ws.length; j++) {
                    weight[i][j] = Double.valueOf(ws[j]);
                }
                i++;
            }
        } catch (IOException ex) {
            Logger.getLogger(WeightParser.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
}
