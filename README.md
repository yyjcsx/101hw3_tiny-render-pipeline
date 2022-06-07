# 101hw3_tiny-render-pipeline
## 使用说明
本程序是Games101课程的作业3，实现了一个小的渲染管线。编译后通过传入不同的参数可使用不同的shader，如
```
./Rasterizer output.png texture 
```
即为使用代码中的 texture shader  
同理，可选的参数有texture,phong,bump,displacement.  
运行结果如下  
实现Blinn-Phong  
![output](https://user-images.githubusercontent.com/50654768/172326930-aa18cd4b-db91-4c47-acee-3a6fe83f486e.png)  
实现纹理贴图  
![output_tex](https://user-images.githubusercontent.com/50654768/172327173-3a1d8c65-2f11-4694-bc4a-cace0316da09.png)  
实现凹凸贴图  
![output_bump](https://user-images.githubusercontent.com/50654768/172327226-6043090b-56ea-4485-85d0-3b67c70e9da4.png)    
实现移位贴图displacement mapping  
![output_displacement](https://user-images.githubusercontent.com/50654768/172327430-5ec73fec-077f-4b9f-ada0-5bf9ad5f0bf3.png)    

# 存在的问题及修复情况：
## z坐标正负颠倒
   框架中的z坐标是遵循越大越远，越小越近的，和PPT和虎书上叙述不符  
   修改方法：  
      1.将get_projection_martix的
      ```
      float t = zNear*tan(eye_fov/2*MY_PI/180);
      ```
      改为
      ```
      float t = -zNear*tan(eye_fov/2*MY_PI/180);
      ```
      2.光栅化作zbuffer时遵循越小越远，越大越近的原则，如果z插值结果大于depth buffer则应予以保留
      3.rasterizer.cpp中rasterizer::clear函数作深度缓冲初始化时，将初始值设为负无穷(最远)，而不是正无穷。
## rasterizerize_triangle透视投影矫正不起作用
   这里由于toVector4()的存在，t的w坐标永远为1，所以相当于没进行透视投影矫正，按理说经过mvp变换后的坐标w值应该加以利用，但是在上一个draw函数里面提前归一化了。如果想改的话应该从上一个draw函数开始改，由于时间原因我尚未修改，修改之后会做相应更新
