# kaggle-dogs-classification
语言：Python3.6
GPU：Nvidia Titan X
框架：Mxnet

训练集10222张, 测试集10357张, 
整理好的数据集下载地址：链接：https://pan.baidu.com/s/1H19DPiCN-oycE2PtCgEdbg 
提取码：8fqc 

执行main.ipynb即可生成提交文件


**结果分析**

我们在实验里采用了控制变量法，控制总训练轮数为100，并以提交到kaggle上的测试结果作为评估标准，对比不同条件下每种网络的表现。每个网络都用ImageNet上的预训练权重进行初始化,参数除表格中已有注明以外其他条件均全部保持一致。

*对比一: 网络深度对模型表现的影响：*

| Model | ResNet 50 | ResNet 101 | ResNet 152 |
| ------------- | ------------- | ------------- | ------------- |
| Scores | 0.33407| 0.25398 | 0.18447 |
|Ranking | 519 | 417 | 204|

由以上表格中的数据我们可以看出网络深度对测试成绩影响效果明显，同样的条件下越深的网络提高测试成绩的效果越好。

*对比二：两种学习衰减算法：*（1）每训练10个epoch，将学习率乘以0.1，我们称之为steps decay
（2）当验证集上loss在后面的5个epoch内未降低时，将学习率乘0.1，我们称之为Threshold decay

| Model | Steps decay | Rank | Threshold decay|Rank|
| ------------- | ------------- | ------------- | ------------- | ------------- |
|ResNet 101| 0.25398| 417 | 0.24626|402|
|ResNet 152 | 0.18447 |204| 0.17783|194|

由上表格我们可以看出相比而言Threshold decay效果更好，更有利于找到“最优解”。

*对比三：不同集成方式对网络表现的影响：*

<table border=0 cellpadding=0 cellspacing=0 width=441 style='border-collapse:
 collapse;table-layout:fixed;width:332pt'>
 <col width=101 style='mso-width-source:userset;mso-width-alt:3242;width:76pt'>
 <col width=85 span=4 style='mso-width-source:userset;mso-width-alt:2709;
 width:64pt'>
 <tr class=xl658911 height=19 style='height:14.5pt'>
  <td height=19 class=xl728911 width=101 style='height:14.5pt;width:76pt'><span
  lang=EN-US>Model</span></td>
  <td class=xl738911 width=85 style='width:64pt'><span lang=EN-US>ResNet 50</span></td>
  <td class=xl738911 width=85 style='width:64pt'><span lang=EN-US>ResNet 101</span></td>
  <td class=xl738911 width=85 style='width:64pt'><span lang=EN-US>ResNet 152</span></td>
  <td class=xl738911 width=85 style='width:64pt'><span lang=EN-US>Rank</span></td>
 </tr>
 <tr class=xl658911 height=19 style='height:14.5pt'>
  <td height=19 class=xl748911 width=101 style='height:14.5pt;width:76pt'><span
  lang=EN-US>Scores</span></td>
  <td class=xl668911 width=85 style='width:64pt'><span lang=EN-US>0.33407</span></td>
  <td class=xl668911 width=85 style='width:64pt'><span lang=EN-US>0.25398</span></td>
  <td class=xl668911 width=85 style='width:64pt'><span lang=EN-US>0.18447</span></td>
  <td class=xl678911 width=85 style='width:64pt'><span lang=EN-US>　</span></td>
 </tr>
 <tr class=xl658911 height=19 style='height:14.5pt'>
  <td height=19 class=xl748911 width=101 style='height:14.5pt;width:76pt'><span
  lang=EN-US>Soft voting</span></td>
  <td colspan=3 class=xl688911 width=255 style='border-right:1.0pt solid black;
  border-left:none;width:192pt'><span lang=EN-US>0.21211</span></td>
  <td class=xl668911 width=85 style='width:64pt'><span lang=EN-US>297</span></td>
 </tr>
 <tr class=xl658911 height=19 style='height:14.5pt'>
  <td height=19 class=xl748911 width=101 style='height:14.5pt;width:76pt'><span
  lang=EN-US>Soft voting</span></td>
  <td class=xl718911 width=85 style='width:64pt'><span lang=EN-US>　</span></td>
  <td colspan=2 class=xl688911 width=170 style='border-right:1.0pt solid black;
  border-left:none;width:128pt'><span lang=EN-US>0.18943</span></td>
  <td class=xl668911 width=85 style='width:64pt'><span lang=EN-US>216</span></td>
 </tr>
 <tr class=xl658911 height=19 style='height:14.5pt'>
  <td height=19 class=xl748911 width=101 style='height:14.5pt;width:76pt'><span
  lang=EN-US>Weighted</span></td>
  <td colspan=3 class=xl688911 width=255 style='border-right:1.0pt solid black;
  border-left:none;width:192pt'><span lang=EN-US>0.20661</span></td>
  <td class=xl668911 width=85 style='width:64pt'><span lang=EN-US>268</span></td>
 </tr>
 <tr class=xl658911 height=19 style='height:14.5pt'>
  <td height=19 class=xl748911 width=101 style='height:14.5pt;width:76pt'><span
  lang=EN-US>Weighted</span></td>
  <td class=xl678911 width=85 style='width:64pt'><span lang=EN-US>　</span></td>
  <td colspan=2 class=xl688911 width=170 style='border-right:1.0pt solid black;
  border-left:none;width:128pt'><span lang=EN-US>0.18539</span></td>
  <td class=xl668911 width=85 style='width:64pt'><span lang=EN-US>208</span></td>
 </tr>
 <![if supportMisalignedColumns]>
 <tr height=0 style='display:none'>
  <td width=101 style='width:76pt'></td>
  <td width=85 style='width:64pt'></td>
  <td width=85 style='width:64pt'></td>
  <td width=85 style='width:64pt'></td>
  <td width=85 style='width:64pt'></td>
 </tr>
 <![endif]>
</table>

软投票集成并未带来提成，我们推测是因为模型能力差异太大，弱分类器拖低了集成效果。
我们提出的加权投票方法能降低弱分类器在投票过程中所占的权重，提升强分类器的权重，表现优于软投票。未来我们将探索在分类器水平相近的情况下，不同加权方法的优劣。


**结论**

通过这次实验，我们实际应用了细粒度图片识别的方法，加深了对卷积神经网络概念的理解。通过在实验研究不同的学习率衰减策略对网络带来的影响的对比，感受了学习率(步伐)的差异对寻找“最优解”的效果带来的影响。在实验中我们还将监督学习、分类算法、集成学习和数据预处理等课程所学内容付于实践，进一步巩固了课堂所学内容。
