package dataprepare;

import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.UnsupportedEncodingException;

import org.dom4j.Document;
import org.dom4j.DocumentException;
import org.dom4j.io.OutputFormat;
import org.dom4j.io.SAXReader;
import org.dom4j.io.XMLWriter;


public class DomUtil {


  public static Document parseDom(File xmlName) {

    SAXReader saxReader = new SAXReader();
    saxReader.setEncoding("UTF-8");
    try {
      return saxReader.read(xmlName);
    } catch (DocumentException e) {
      e.printStackTrace();
    }
    return null;
  }

  public static Document parseDom(String xmlStr) {
    SAXReader saxReader = new SAXReader();
    saxReader.setEncoding("UTF-8");
    try {
      return saxReader.read(new ByteArrayInputStream(xmlStr.getBytes("UTF-8")));
    } catch (DocumentException e) {
      e.printStackTrace();
    } catch (UnsupportedEncodingException e) {
      e.printStackTrace();
    }
    return null;
  }


  public static void writeOut(String filename, Document doc) {

    try {
      OutputFormat outFmt = new OutputFormat("\t", true);
      outFmt.setEncoding("UTF-8");

      XMLWriter output = new XMLWriter(new FileOutputStream(filename), outFmt);
      output.write(doc);
      output.close();
    } catch (IOException e) {
      System.out.println(e.getMessage());
    }

  }

  public static void main(String[] args) {
    // TODO Auto-generated method stub
    Document doc = DomUtil.parseDom(new File("kaiqi_themes.xml"));
    System.out.println(doc.asXML());
  }

}
