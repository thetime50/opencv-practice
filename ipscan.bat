:: https://jingyan.baidu.com/article/27fa7326a05fd246f8271f9a.html
@echo off
color F0
:: ���ô��ڱ���ɫΪ��ɫ��������ɫΪ��ɫ
title ������ɨ����(By TaoGe)
:: ���ô��ڱ���
echo.
echo ������Ҫɨ���IP�Σ�ֱ�Ӱ��س���Ϊ192.168.1��
set /p IpDuan=
:: ���û����븳ֵ��IpDuan����
if "%IpDuan%"=="" (set IpDuan=192.168.1)
:: �ж�IpDuan�����Ƿ�ֵ�����Ϊ�գ���ֵΪ192.168.1
echo ������Ҫɨ���IP��ʼλ��ֱ�Ӱ��س���Ϊ1��
set /p QiShi=
:: ���û����븳ֵ��QiShi����
if "%QiShi%"=="" (set QiShi=1)
:: �ж�QiShi�����Ƿ�ֵ�����Ϊ�գ���ֵΪ1
echo ������Ҫɨ���IP����λ��ֱ�Ӱ��س���Ϊ255��
set /p JieShu=
:: ���û����븳ֵ��JieShu����
if "%JieShu%"=="" (set JieShu=255)
:: �ж�JieShu�����Ƿ�ֵ�����Ϊ�գ���ֵΪ255
echo ��ʼIP��%IpDuan%.%QiShi%  
:: ��ʾ��ʼIP
echo ����IP��%IpDuan%.%JieShu%  
:: ��ʾ����IP
echo ======================================================= >>Ping-%IpDuan%.txt
:: ��¼�ָ���
echo ��ʼʱ�䣺%date%%time% >>Ping-%IpDuan%.txt
:: ��¼��ʼʱ��
echo ��ʼIP��%IpDuan%.%QiShi% >>Ping-%IpDuan%.txt  
:: ��¼��ʼIP
echo ����IP��%IpDuan%.%JieShu% >>Ping-%IpDuan%.txt 
:: ��¼����IP
echo ����ɨ�裬��ȴ�...
echo ��ǰ������ֱ�ӹرմ���
@for /l %%n in (%QiShi%,1,%JieShu%) do @ping -w 600 -n 1 %IpDuan%.%%n|find  /i "ttl" >>Ping-%IpDuan%.txt
:: ��ʼִ��
echo ����ʱ�䣺%date% %time%  >>Ping-%IpDuan%.txt
:: ��¼����ʱ��
echo ======================================================= >>Ping-%IpDuan%.txt
:: ��¼�ָ���
echo ɨ�����,��������˳�...&pause>nul