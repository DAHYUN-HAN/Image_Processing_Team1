from django.shortcuts import render
from .form import ImagePost
from django.utils import timezone
from .models import Image
from django.shortcuts import render, get_object_or_404, redirect
from django.utils import timezone
from .models import Pic

# Create your views here.
from django.http import HttpResponse
from django.template import loader

import cv2
import math
import numpy as np

def order_points(pts):
    rect = np.zeros((4, 2), dtype = "float32")
    
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect
    
def distance(p0, p1):
    return math.sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2)

#사용
def index(request):
    if request.method == 'POST':
        form = ImagePost(request.POST)
        if form.is_valid():
            #모델 객체를 db에 저장하지 않은 상태로 반환
            post = form.save(commit=False)
            post.pub_date = timezone.now()
            post.save()
            for afile in request.FILES.getlist('file'):
                imgub = Pic()
                imgub.fore_image = post
                imgub.image = afile
                imgub.save()
            return redirect('/imagelist#projects')
    else:
        form = ImagePost()
        return render(request, 'index.html', {'form':form})

def imagelist(request):
    image = Image.objects.last()
    for pic in image.pic_set.all():
        url = pic.image.url
        img = cv2.imread(url[1:])
        img_ori = img.copy()
        img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        cv2.imwrite('coinapp/static/coinapp/result/images/img_gray.jpg', img_gray)
        # Blurring
        dst = cv2.medianBlur(img_gray, 9)  #medianBlur
        cv2.imwrite('coinapp/static/coinapp/result/images/dst.jpg', dst)
        dst2 = cv2.bilateralFilter(dst, 9, 100, 100) # Bilateral Filtering
        cv2.imwrite('coinapp/static/coinapp/result/images/dst2.jpg', dst2)
        binary_img = cv2.adaptiveThreshold(dst2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 33, 5)
        cv2.imwrite('coinapp/static/coinapp/result/images/binary_img.jpg', binary_img)
        
        img, contour, hierarchy = cv2.findContours(binary_img, 
                                            cv2.RETR_EXTERNAL, 
                                            cv2.CHAIN_APPROX_SIMPLE)
        img = img_ori.copy()
        #추출된 contour 중 가장 큰 contour 찾기
        max = 0;
        index = -1;
        for i in range(len(contour)):
           # print(max,i)
            if(max<len(contour[i])):        
                index = i
                max = len(contour[i])

        cnt = contour[index]
        epsilon1 = 0.1*cv2.arcLength(cnt, True)
        approx1 = cv2.approxPolyDP(cnt, epsilon1, True)
        cv2.drawContours(img, [approx1], 0, (0, 255, 0), 20)
        
        #만약 정상적으로 4개의 점을 찾지 못하면 threshold를 이용하여 재시도
        if (len(approx1)!=4):
            ret_threshold, binary_img = cv2.threshold(dst2,
                                                  170,
                                                  255,
                                                  cv2.THRESH_BINARY)

            img, contour, hierarchy = cv2.findContours(binary_img, 
                                                    cv2.RETR_EXTERNAL, 
                                                    cv2.CHAIN_APPROX_SIMPLE)

            img = img_ori.copy()   

            max = 0;
            index = -1;
            for i in range(len(contour)):
           # print(max,i)         if(max<len(contour[i])):        
                    index = i
                    max = len(contour[i])
                    cnt=contour[index]

            cv2.drawContours(img, [cnt], 0, (255, 255, 0), 1)

            epsilon1 = 0.1*cv2.arcLength(cnt, True)
            approx1 = cv2.approxPolyDP(cnt, epsilon1, True)

            cv2.drawContours(img, [approx1], 0, (0, 255, 0), 20)
        else:
            pass 
        
        cv2.imwrite('coinapp/static/coinapp/result/images/drawContoursimg.jpg', img)
        
        #정확하게 A4를 perspect하도록 좌표 재조정
        approx_list = []
        for i in range(8):
            vals = approx1.item(i)
            approx_list.append(vals)

        approx_list = np.array(approx_list)
        pts1 = approx_list.reshape(4,-1).astype(np.float32)
        
        pts1 = order_points(pts1)
        
        dist0_1 = distance(pts1[0], pts1[1])
        dist1_2 = distance(pts1[1], pts1[2])

        if(dist0_1 > dist1_2):
            pts2 = np.float32([[0,0],[1189,0],[1189,841],[0,841]]) # 이 상태는 좌측하단, 우측상단, 우측하단, 좌측상단임
        elif(dist0_1 < dist1_2):
            pts2 = np.float32([[1189,0],[1189,841],[0,841], [0,0]]) # 이 상태는 좌측상단, 우측상단, 우측하단, 좌측하단임 

        #Perspective Transform 
        M = cv2.getPerspectiveTransform(pts1, pts2)
        dst = cv2.warpPerspective(img, M, (1189, 841))    
        cv2.imwrite('coinapp/static/coinapp/result/images/warpPerspectiveimg.jpg', dst)
        
        #원의 윤곽 부각되도록
        gray = cv2.cvtColor(dst,cv2.COLOR_BGR2GRAY)
        cv2.imwrite('coinapp/static/coinapp/result/images/circlegray.jpg', gray)
        blur = cv2.medianBlur(gray, 5)
        cv2.imwrite('coinapp/static/coinapp/result/images/circleblur.jpg', blur)
        line = cv2.bilateralFilter(blur, 5, 30, 30) # Bilateral Filtering
        cv2.imwrite('coinapp/static/coinapp/result/images/circleline.jpg', line)
        cimg = dst.copy()
        
        cimg = dst.copy()
        
        # 겹친 동전의 모양까지 원으로 검출하는 기준을 정의한다
        circles = cv2.HoughCircles(line, cv2.HOUGH_GRADIENT, 1, 20, np.array([]), param1=150, param2=20, minRadius=15, maxRadius=50)
        circles = np.uint16(np.around(circles))
        
        fivehund = 0
        onehund = 0
        fifty = 0
        ten = 0

        # 위 조건에 맞춰 cimg에 동전별로 원을 그린다
        for circ in circles[0,:]:
            if circ[2] >= 50: #or circ[2] >= 38 or circ[2] >= 28:
                fivehund += 1
                cv2.circle(cimg, (circ[0], circ[1]), circ[2], (0,255,0) ,2)
            elif circ[2] >= 45: #or circ[2] >= 35 or circ[2] >= 25:
                onehund += 1
                cv2.circle(cimg, (circ[0], circ[1]), circ[2], (0,0,255) ,2)
            elif circ[2] >= 40: #or circ[2] >= 22:
                fifty += 1
                cv2.circle(cimg, (circ[0], circ[1]), circ[2], (0,0,0) ,2)
            else:
                ten += 1
                cv2.circle(cimg, (circ[0], circ[1]), circ[2], (255,0,0), 2)
        cv2.imwrite('coinapp/static/coinapp/result/images/cimg.jpg', cimg)
        
        # 동전 좌표 확인하기
        count = circles[0,:]        # 인식된 원의 개수를 count라고 지정한다
        # len(count)
        c_array = circles.reshape(len(count), -1)                      # 검출된 원의 개수를 행의 개수로 하는 배열을 array라고 지정한다
        radiusSet = c_array[:,-1]


        # 총액 계산하기
        total = (500 * fivehund + 100 * onehund + 50 * fifty + 10 * ten)
        
    return render(request, 'imagelist.html', {'image':image, 'fivehund':fivehund, 'onehund':onehund, 'fifty':fifty, 'ten':ten, 'total':total})
