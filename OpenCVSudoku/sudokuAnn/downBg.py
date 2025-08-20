import requests
import os
import time
import random
from urllib.parse import quote
from PIL import Image
import io
import cloudscraper
import re

from multiprocessing import Pool, Manager, Lock
import json
from datetime import datetime
import hashlib

def string_to_md5(text):
    """将字符串转换为MD5哈希值"""
    # 创建MD5对象
    md5_hash = hashlib.md5()
    # 更新哈希对象（需要将字符串编码为bytes）
    md5_hash.update(text.encode('utf-8'))
    # 获取十六进制哈希值
    return md5_hash.hexdigest()
import multiprocessing as mp
from multiprocessing import Pool, set_start_method
import os
import cloudscraper
from PIL import Image
import io
import re
from urllib.parse import quote
import hashlib
import time
import random

# 设置启动方法
try:
    set_start_method('spawn', force=True)
except RuntimeError:
    pass

def download_worker(task_data):
    """独立的工作函数"""
    img_url, query, width, height, save_dir,err_domains, index = task_data
    
    try:
        domain = img_url.split('/')[2]
        max_err_count = 3
        if domain in err_domains and err_domains[domain] >= max_err_count:
            print(f"跳过高错误率域名: {domain} cnt:{err_domains[domain]}/{max_err_count}")
            err_domains[domain] = err_domains.get(domain, 0) + 1
            return {'status': 'failed', 'error': f'跳过高错误率域名 cnt:{err_domains[domain]}/{max_err_count} for {img_url} '}
        # 使用MD5避免文件名冲突
        url_hash = hashlib.md5(img_url.encode()).hexdigest()[:8]
        filename = f"{query}_{width}x{height}_{url_hash}.jpg"
        filepath = os.path.join(save_dir, filename)
        if (index+1) % 50 == 0:
            print(f"进度: {index+1} 下载中... {filename}")
        # 跳过已存在的文件
        if os.path.exists(filepath):
            print(f"文件已存在: {filepath}")
            return {'status': 'skipped', 'filename': filename}
        
        scraper = cloudscraper.create_scraper()
        proxies = {
            'http': 'socks5://127.0.0.1:10808',
            'https': 'socks5://127.0.0.1:10808'
        }
        scraper.headers.update({
            'User-Agent': random.choice([
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
            ])
        })
        
        img_response = scraper.get(img_url, timeout=15,proxies=proxies)
        
        if img_response.status_code == 200:
            img = Image.open(io.BytesIO(img_response.content))
            if img.size != (width, height):
                img = img.resize((width, height), Image.Resampling.LANCZOS)
            
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            img.save(filepath, 'JPEG', quality=90)
            time.sleep(random.uniform(0.5, 2))
            
            return {'status': 'success', 'filename': filename}
        else:
            err_domains[domain] = err_domains.get(domain, 0) + 1
            return {'status': 'failed', 'error': f'HTTP {img_response.status_code} for {img_url}'}
            
    except Exception as e:
        err_domains[domain] = err_domains.get(domain, 0) + 1
        return {'status': 'failed', 'error': str(e) + f' for {img_url}'}

class ParallelImageDownloader:
    def __init__(self, max_processes=None):
        self.max_processes = max_processes or min(mp.cpu_count(), 6)
        self.manager = Manager()
        self.err_domains = self.manager.dict()  # 错误域名统计
    
    def get_image_urls(self, query, count, max_pages=0):
        """更智能的滚动加载，支持多种分页策略"""
        if not max_pages or max_pages <= 0:
            max_pages = (count // 35) * 1.3
        try:
            scraper = cloudscraper.create_scraper()
            query_encoded = quote(query)
            all_image_urls = set()  # 使用集合自动去重
            page = 1
            page_count = 35
            zero_url_count = 0
            
            print(f"🔍 搜索: '{query}'，目标: {count} 张图片")
            
            while len(all_image_urls) < count and page <= max_pages:
                # 多种URL格式尝试
                urls_to_try = [
                    f"https://www.bing.com/images/search?q={query_encoded}&first={(page-1)*page_count}&count={page_count}",
                    f"https://www.bing.com/images/search?q={query_encoded}&form=HDRSC2&first={(page-1)*page_count}&count={page_count}",
                    f"https://www.bing.com/images/search?q={query_encoded}&qs=HS&form=QBIR&first={(page-1)*page_count}&count={page_count}"
                ]
                
                page_success = False
                for url in urls_to_try:
                    try:
                        response = scraper.get(url, timeout=15)
                        if response.status_code != 200:
                            continue
                        
                        # 多种匹配模式
                        patterns = [
                            r'murl&quot;:&quot;(https?://[^&]+?\.(?:jpg|jpeg|png|webp))',
                            r'src=&quot;(https?://[^&]+?\.(?:jpg|jpeg|png|webp))',
                            r'imgurl&quot;:&quot;(https?://[^&]+?\.(?:jpg|jpeg|png|webp))'
                        ]
                        
                        page_urls = set()
                        for pattern in patterns:
                            found_urls = re.findall(pattern, response.text)
                            page_urls.update(found_urls)
                        
                        if page_urls:
                            new_count = len(page_urls - all_image_urls)
                            all_image_urls.update(page_urls)
                            if new_count == 0:
                                zero_url_count += 1
                            else:
                                zero_url_count = 0
                            
                            print(f"📄 第 {page} 页找到 {new_count} 张新图片，总计: {len(all_image_urls)}")
                            page_success = True
                            break
                            
                    except Exception as e:
                        continue
                if zero_url_count >= 4:
                    print(f"⚠️  第 {page} 页没有新图片，可能是重复内容")
                    break
                
                if not page_success:
                    print(f"⚠️  第 {page} 页加载失败")
                
                # 如果已经达到目标，提前退出
                if len(all_image_urls) >= count:
                    break
                
                page += 1
                time.sleep(random.uniform(0.5, 3))  # 随机延迟
            
            result = list(all_image_urls)[:count]
            print(f"🎯 搜索完成: 找到 {len(result)} 张图片")
            return result
            
        except Exception as e:
            print(f"❌ 搜索失败: {e}")
            return []
    
    def download_parallel(self, query, count=20, width=512, height=512, save_dir='downloads'):
        """主下载方法"""
        os.makedirs(save_dir, exist_ok=True)
        
        # 获取图片URL
        image_urls = self.get_image_urls(query, count)
        if not image_urls:
            print("❌ 未找到图片链接")
            return {'success': 0, 'failed': count}
        
        print(f"✅ 找到 {len(image_urls)} 个图片链接")
        
        # 准备任务
        tasks = []
        for i, img_url in enumerate(image_urls[:count]):
            tasks.append((img_url, query, width, height, save_dir,self.err_domains, i))
        
        # 使用进程池
        success_count = 0
        failed_count = 0
        
        with Pool(processes=self.max_processes) as pool:
            results = pool.map(download_worker, tasks)
            
            for result in results:
                if result['status'] == 'success' or result['status'] == 'skipped':
                    success_count += 1
                    # print(f"✅ 成功: {result['filename']}")
                else:
                    failed_count += 1
                    print(f"❌ 失败: {result['error']}")
        
        return {'success': success_count, 'failed': failed_count}



# 基本使用
if __name__ == "__main__":
    
    # 方法3: 使用类管理

    downloader = ParallelImageDownloader(max_processes=4)
    save_dir = "./img/bg"
    # query="background book"
    # # query="background 书桌"
    # stats = downloader.download_parallel(
    #     query=query,
    #     count=3000,
    #     width=1920,
    #     height=1080,
    #     save_dir=save_dir
    # )
    # query="background 书桌"
    # stats = downloader.download_parallel(
    #     query=query,
    #     count=3000,
    #     width=1920,
    #     height=1080,
    #     save_dir=save_dir
    # )
    # query="background city"
    # stats = downloader.download_parallel(
    #     query=query,
    #     count=3000,
    #     width=1920,
    #     height=1080,
    #     save_dir=save_dir
    # )
    
    # query="background paper"
    # stats = downloader.download_parallel(
    #     query=query,
    #     count=3000,
    #     width=1920,
    #     height=1080,
    #     save_dir=save_dir
    # )
    # query="background scenery"
    # stats = downloader.download_parallel(
    #     query=query,
    #     count=3000,
    #     width=1920,
    #     height=1080,
    #     save_dir=save_dir
    # )
    # query="background room"
    # stats = downloader.download_parallel(
    #     query=query,
    #     count=3000,
    #     width=1920,
    #     height=1080,
    #     save_dir=save_dir
    # )
    query="background 报纸"
    stats = downloader.download_parallel(
        query=query,
        count=3000,
        width=1920,
        height=1080,
        save_dir=save_dir
    )
    print(f"统计信息: {stats}")