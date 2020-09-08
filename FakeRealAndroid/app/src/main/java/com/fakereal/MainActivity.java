package com.fakereal;

import android.app.Activity;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.net.Uri;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.provider.MediaStore;
import android.widget.ImageView;
import android.widget.Toast;

import java.net.Socket;
import java.io.PrintWriter;
import java.io.BufferedReader;
import java.io.InputStreamReader;
public class MainActivity extends Activity {
    private static final int PICK_IMAGE = 100;
    Uri imageUri;
    EditText ip;
    ImageView iv;
    StringBuilder sb = new StringBuilder();
    String message;
    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        initClickListner();
    }
    private void initClickListner()	{
        ip = (EditText)findViewById(R.id.ip);
        iv = (ImageView)findViewById(R.id.imgView);
        Button admin = (Button) findViewById(R.id.upload);
        admin.setOnClickListener(new Button.OnClickListener() {
            public void onClick(View v) {
                Intent gallery = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.INTERNAL_CONTENT_URI);
                startActivityForResult(gallery, PICK_IMAGE);
            }
        });


    }
    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data){
        super.onActivityResult(requestCode, resultCode, data);
        if (resultCode == RESULT_OK && requestCode == PICK_IMAGE){
            try {

                imageUri = data.getData();
                Bitmap bitmap = MediaStore.Images.Media.getBitmap(this.getContentResolver(), imageUri);
                iv.setImageBitmap(bitmap);
                bitmap = Bitmap.createScaledBitmap(bitmap, 64, 64, true);
                int height = bitmap.getHeight();
                int width = bitmap.getWidth();
                sb.delete(0,sb.length());
                sb.append(width+"#"+height+"#");
                for (int i = 0; i < width; i++) {
                    for (int j = 0; j < height; j++) {
                        int pixel = bitmap.getPixel(i, j);
                        int red = Color.red(pixel);
                        int green = Color.green(pixel);
                        int blue = Color.blue(pixel);
                        sb.append(red+"?"+green+"?"+blue+",");
                    }
                    sb.deleteCharAt(sb.length()-1);
                    sb.append("#");
                }
                sb.append("hello");
                System.out.println(sb.toString());
                System.out.println(sb.toString().length());
                Runnable r = new Runnable(){
                    public void run(){
                        sendRequest();
                    }
                };
                Thread th = new Thread(r);
                th.start();
                th.join();
                Toast.makeText(MainActivity.this, message, Toast.LENGTH_LONG).show();
            }catch(Exception e){
                e.printStackTrace();
            }
        }
    }
    public String sendRequest(){
        String msg = "error in getting output";
        try{
            String s1 = ip.getText().toString().trim();
            Socket socket = new Socket(s1,5000);
            PrintWriter pw = new PrintWriter(socket.getOutputStream(),true);
            pw.println(sb.toString());
            pw.flush();
            BufferedReader br = new BufferedReader(new InputStreamReader(socket.getInputStream()));
            msg = br.readLine();
            message = msg;
        }catch(Exception e){
            e.printStackTrace();
        }
        return msg;
    }
}

