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
public class Node {
    public ArrayList<Node>  nextNodes, prevNodes;
    String id;
    
    public Node(ArrayList<Node> prev, ArrayList<Node> next) {
        nextNodes = next;
        prevNodes = prev;
    }
    public Node(String id, ArrayList<Node> prev, ArrayList<Node> next) {
        this.id = id;
        nextNodes = next;
        prevNodes = prev;
    }

    public void setID(String id) {
        this.id = id;
    }
    
    public String getID() {
        return id;
    }    
    public ArrayList<Node> getNextNodes() {
        return nextNodes;
    }    
    public ArrayList<Node> getPrevNodes() {
        return prevNodes;
    }
    
    
}
