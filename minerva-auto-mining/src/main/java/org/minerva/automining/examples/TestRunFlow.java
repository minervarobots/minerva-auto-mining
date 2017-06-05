

package org.minerva.automining.examples;



import java.io.File;
import weka.knowledgeflow.*;
import weka.core.WekaPackageManager;
import weka.core.Settings;
import weka.core.Utils;

public class TestRunFlow {

    public static void main(String[] args) {
	try {
	    // initialize global Weka logging (i.e. to ~/wekafiles/weka.log)
	    weka.core.logging.Logger.log(weka.core.logging.Logger.Level.INFO,
					 "Logging started");

	    // make sure all plugin packages are loaded
	    WekaPackageManager.loadPackages(false, true, false);

	    // initialize a new FlowRunner object
	    FlowRunner fr = new FlowRunner();

	    Settings settings = new Settings("weka", KFDefaults.APP_ID);
	    settings.loadSettings();
	    settings.applyDefaults(new KFDefaults());
	    fr.setSettings(settings);

	    // .kf file to execute in args[0]
//	    String fileName = args[0];
			    
//	    args[0] = "";

	    String fileName = "kf/kf-example2.kf";
		//  String fileName = "kf/kf-example2.kf";
	    
	  //String fileName = "kf/kf-example_find_error.kf";
	    
	    
	    // Load a flow and configure the FlowRunner with it
	    Flow flowToRun = Flow.loadFlow(new File(fileName), new FlowRunner.SimpleLogger());
	    fr.setFlow(flowToRun);
	    
	    // run the flow and wait until it is finished
	    fr.run();
	    fr.waitUntilFinished();
	    System.exit(0);
	    System.out.println("KF Done");
	} catch (Exception ex) {
	    ex.printStackTrace();
	}
    }
}
