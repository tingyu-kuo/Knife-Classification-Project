## Knife-Classification-Project

計畫名稱: 基於深度學習的刀具分類與分析  
計畫編號: MOST 108-2221-E-194-057-MY3  
計畫內容: 以深度學習方法對刀具拓樸圖進行磨耗程度分類  
<br>
實作方法:  
	Q1. 因資料量不足，做普通的擴增方法易破壞拓樸圖中的物理特性。  
		&emsp;Ans. 將拓樸圖以點雲的方式呈現，較容易進行擴增。  
	Q2. 因拓樸圖大小為1000x1000，即點的個數有100萬，現有硬體不易實現。  
		&emsp;Ans. 利用random sample的方式，由每個sliding windows中，隨機取出top n的數值，  
		&emsp;&emsp;&emsp;n為可調參數，不僅能減少點的數量，也能由隨機性來進行擴增。  
	ToDo: 再配合3D空間旋轉進行資料擴增，以Pointnet++等相關模型來做分類測試。
</br>
### 刀具拓樸圖(2D & 3D):
<img src="https://github.com/tingyu-kuo/Knife-Classification-Project/blob/main/images/plot_2d.PNG" width="500"/><br/>
<img src="https://github.com/tingyu-kuo/Knife-Classification-Project/blob/main/images/plot_3d.PNG" width="500"/><br/>
### 刀具拓樸深度圖3D:
<img src="https://github.com/tingyu-kuo/Knife-Classification-Project/blob/main/images/depth_3d.PNG" width="500"/><br/>
### 刀具拓樸深度圖3D(隨機採樣 & Resize):
#### 將原先拓樸圖1000x1000調整為100x100，且因隨機性，每次執行結果都會有些微不同，達到擴增的目的
<img src="https://github.com/tingyu-kuo/Knife-Classification-Project/blob/main/images/depth_random.PNG" width="500"/><br/>


### Testing Model Performance on Another Dataset 
#### Main Architecture
<img src="https://github.com/tingyu-kuo/Knife-Classification-Project/blob/main/images/figure1.png" width="500"/><br/>
#### DACL Attention Method
<img src="https://github.com/tingyu-kuo/Knife-Classification-Project/blob/main/images/figure2.png" width="500"/><br/>
#### Self-attention Method
<img src="https://github.com/tingyu-kuo/Knife-Classification-Project/blob/main/images/figure3.png" width="500"/><br/>
#### Compare the Result Between Two Methods and Baseline
<img src="https://github.com/tingyu-kuo/Knife-Classification-Project/blob/main/images/table1.png" width="500"/><br/>
#### Confusion metrices of Two Methods and Baseline
<img src="https://github.com/tingyu-kuo/Knife-Classification-Project/blob/main/images/figure4.png" width="500"/><br/>
