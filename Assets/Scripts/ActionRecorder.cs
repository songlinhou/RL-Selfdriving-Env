using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;
using System.IO;

public class ActionRecorder : MonoBehaviour
{
    // Start is called before the first frame update
    public static Dictionary<string, string> CONFIG = new Dictionary<string, string>();
    private RoadLine roadLine;
    private string startPoint;
    private string endPoint;
    void Start()
    {
        string wd = Directory.GetCurrentDirectory();
        print("wd=" + wd);
        readConfig();
        roadLine = GetComponent<RoadLine>();
        startPoint = roadLine.startPoint.name;
        endPoint = roadLine.endPoint.name;
    }

    // Update is called once per frame
    void Update()
    {

    }

    void readConfig() {
        int counter = 0;
        string line;

        // Read the file and display it line by line.  
        System.IO.StreamReader file = new System.IO.StreamReader(@"config.txt");
        while ((line = file.ReadLine()) != null)
        {
            print(line);
            parseConfigSentence(line);
            counter++;
        }

        file.Close();
    }

    public void writeRecordsToFile(List<string> records) {
        StreamWriter writer = File.AppendText("records.txt");
        foreach (string record in records) {
            writer.WriteLine(record);
        }
        writer.Flush();
        writer.Close();
        print("records are saved.");
    }

    public void writeCSVRecordsToFile(List<string> featureNames, List<string> records) {
        string fileName = "record_data(" + startPoint + "_" + endPoint + ").csv";
        StreamWriter writer;
        if (!File.Exists(fileName)) {
            String featureRow = "";
            foreach (string feat in featureNames)
            {
                featureRow += (feat + ",");
            }
            writer = File.AppendText(fileName);
            writer.WriteLine(featureRow.Substring(0, featureRow.Length - 1));
            writer.Flush();
            writer.Close();
        }
        StreamWriter writer2 = File.AppendText(fileName);
        
        foreach (string record in records)
        {
            writer2.WriteLine(record);
        }
        writer2.Flush();
        writer2.Close();
        print("csv records are saved to " + fileName + ".");
    }

    void parseConfigSentence(string sentence) {
        if (sentence.StartsWith("#")) {
            return;
        }
        if (sentence.Contains("#")) {
            sentence = sentence.Substring(0, sentence.IndexOf("#"));
        }
        if (sentence.Contains("=")) {
            string[] subs = sentence.Split('=');
            CONFIG[subs[0].Trim()] = subs[1].Trim();
            print("add new config: " + subs[0] + " = " + subs[1]);
        }
    }
}
