import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.math.BigInteger;
import java.util.ArrayList;

import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
 
/**
 * @author Saif Lakhani
 */
 

 public class Process{
	static ArrayList<String> takenIds = new ArrayList<>();
	static ArrayList<String> foundFourList = new ArrayList<>();
	public static ArrayList<JSONObject> finalData = new ArrayList<>();
	static int validData = -1;
	static JSONArray finalArray = new JSONArray();
    public static void main(String[] args) throws IOException {
        JSONParser parser = new JSONParser();
        try {
            Object obj = parser.parse(new FileReader(
                    "/Users/saif/Documents/sigmaFinale.json"));
            JSONArray jsonArray = (JSONArray) obj;
            System.out.println(jsonArray.size());
            
            for(int i=0;i<jsonArray.size();i++) {
            	JSONObject currentObject = (JSONObject) jsonArray.get(i);
            	try {
            	String binAddr = Integer.toBinaryString(Integer.parseInt((String) currentObject.get("rssiVal"), 16));
            	binAddr = twosCompliment(binAddr);
            	int foo = 0;
            	foo = Integer.parseInt(binAddr, 2);
            	currentObject.put("rssi", foo);
            	finalArray.add(currentObject);
            	//System.out.println(foo);
            	if(i%10000==0) {
            		writeToFile();
            		System.out.println("Written");
            		finalArray.clear();
            	}
            	}catch(Exception e) {
            		continue;
            	}
            	
            }
            writeToFile();
            
        } catch (Exception e) {
            e.printStackTrace();
            //writeToFile();
        }
        
        
}
    static void writeToFile() throws IOException {
    	try (FileWriter file = new FileWriter("/Users/saif/Documents/meow.json",true)) {
    		BufferedWriter writer = new BufferedWriter(file);
			writer.write(finalArray.toString());
			writer.flush();
			writer.close();
			System.out.println("Successfully Copied JSON Object to File...");
			//System.out.println("\nJSON Object: " + finalArray);
		}
    }
	
	
	  public static String twosCompliment(String bin) {
	        String twos = "", ones = "";

	        for (int i = 0; i < bin.length(); i++) {
	            ones += flip(bin.charAt(i));
	        }
	        int number0 = Integer.parseInt(ones, 2);
	        StringBuilder builder = new StringBuilder(ones);
	        boolean b = false;
	        for (int i = ones.length() - 1; i > 0; i--) {
	            if (ones.charAt(i) == '1') {
	                builder.setCharAt(i, '0');
	            } else {
	                builder.setCharAt(i, '1');
	                b = true;
	                break;
	            }
	        }
	        try {
	        if (!b)
	            builder.append("1", 0, 7);
	        }catch(IndexOutOfBoundsException e) {
	        	return "11111";
	        	//e.printStackTrace();
	        }

	        twos = builder.toString();

	        return twos;
	    }

	// Returns '0' for '1' and '1' for '0'
	    public static char flip(char c) {
	        return (c == '0') ? '1' : '0';
	    }

 }
