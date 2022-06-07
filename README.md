# 101hw3_tiny-render-pipeline
## 使用说明
本程序是Games101课程的作业3，实现了一个小的渲染管线。编译后通过传入不同的参数可使用不同的shader，如  
  ./Rasterizer output.png texture
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

## 存在的问题及修复情况：
# z坐标正负颠倒
