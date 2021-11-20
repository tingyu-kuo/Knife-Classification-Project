## Knife-Classification-Project

計畫名稱: 基於深度學習的刀具分類與分析<br>
計畫編號: MOST 108-2221-E-194-057-MY3 <br>
計畫內容: 以深度學習方法對刀具拓樸圖進行磨耗程度分類<br>
實作方法:<br>
	Q1. 因資料量不足，做普通的擴增方法易破壞拓樸圖中的物理特性。<br>
		<&nbsp>Ans. 將拓樸圖以點雲的方式呈現，較容易進行擴增。<br>
	Q2. 因拓樸圖大小為1000x1000，即點的個數有100萬，現有硬體不易實現。<br>
		&nbspAns. 利用random sample的方式，由每個sliding windows中，隨機取出top n的數值，n為可調參數，<br>
			 &nbsp&nbsp不僅能減少點的數量，也能由隨機性來進行擴增。<br>
	ToDo: 再配合3D空間旋轉進行資料擴增，以Pointnet++等相關模型來做分類測試。