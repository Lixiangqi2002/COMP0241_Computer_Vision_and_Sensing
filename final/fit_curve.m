ptCloud = pcread("/home/selina-xiangqi/ucl2024/ComputerVison&Sensing/COMP0241_FINAL_G10/Dataset/rtabmap_pointCloud/cloud_02_segmentation.ply");
pointscolor=uint8(zeros(ptCloud.Count,3));
pointscolor(:,1)=255;
pointscolor(:,2)=255;
pointscolor(:,3)=51;
ptCloud.Color=pointscolor;
%figure
%pcviewer(ptCloud)
maxDistance =  1;
roi = [-inf,0.5;0.2,0.4;0.1,inf];
sampleIndices = findPointsInROI(ptCloud,roi);
[model,inlierIndices] = pcfitsphere(ptCloud,maxDistance,SampleIndices=sampleIndices);
globe = select(ptCloud,inlierIndices);
%figure
%pcviewer(globe)
%pcwrite(globe,"/home/selina-xiangqi/ucl2024/ComputerVison&Sensing/COMP0241_FINAL_G10/Dataset/rtabmap_pointCloud/cloud_02_segmentation_sphere.ply");