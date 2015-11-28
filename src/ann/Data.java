/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package ann;

import java.util.ArrayList;

/**
 *
 * @author khaidzir
 */
public class Data {
    
    public ArrayList<Double> input, target;
    
    Data() {
        input = new ArrayList<> ();
        target = new ArrayList<> ();
    }
    
    Data(ArrayList<Double> _input, ArrayList<Double> _target) {
        input = _input;
        target = _target;
    }
    
}
