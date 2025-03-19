from fastapi import FastAPI, Request, HTTPException
import base64
import numpy as np
import logging
import os
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from SPC.CControlChart import CControlChart
from SPC.Rule import Rule
from SPC.Rule01 import Rule01
from SPC.Rule02 import Rule02
from SPC.Rule03 import Rule03
from SPC.Rule04 import Rule04
from SPC.Rule05 import Rule05
from SPC.Rule06 import Rule06
from SPC.Rule07 import Rule07
from SPC.Rule08 import Rule08

import base64
import asyncio
import logging
import os
from datetime import datetime
# 配置日志
logging.basicConfig(
    filename='Cchart.log',
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

app = FastAPI()

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def generate_image(data: dict) -> bytes:
    # 创建图表
    data['request_id'] = id(data) # 为请求生成一个唯一 ID 以便追踪
    try:
        data_org = data.get('data', [])
        if not data_org:
            raise HTTPException(status_code=400, detail="Invalid data")

        edit_time_list = [item['edit_time'] for item in data_org]
        data_list = [item['data'] for item in data_org]
        if len(edit_time_list) != len(data_list):
            raise HTTPException(status_code=400, detail="Invalid data")
        chart_title = data.get('chart_title', 'C_Chart')
        rule_list = data.get('rule_list', [])
        upper_line = data.get('upper', 1)
        groupsize = data.get('groupSize', 1)
        chart = CControlChart(data_list=data_list, edit_time_list=edit_time_list, groupSize=groupsize, upper=upper_line,
                              title_name=chart_title)
        chart.dateformat = "%Y-%m-%d %H:%M:%S"
        chart.limits = True
        # 得到对应的判异规则
        if len(rule_list) == 0:
            pass
        else:
            for i in rule_list:
                if i == 1:
                    chart.append_rule(Rule01())
                elif i == 2:
                    chart.append_rule(Rule02())
                elif i == 3:
                    chart.append_rule(Rule03())
                elif i == 4:
                    chart.append_rule(Rule04())
            img = chart.plot()

        return img.getvalue()
    except Exception as e:
        print(e)
    finally:
        pass


@app.post("/api/generate_CChart")
async def generate_chart(request: Request):
    # 确保图片目录存在
    if not os.path.exists('images'):
        os.makedirs('images')

    # 记录访问日志
    client_ip = request.client.host
    logging.info(f"IP: {client_ip} - {request.method} {request.url.path}")

    # 获取请求数据
    try:
        data = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON data")

    # 验证数据
    if not data:
        raise HTTPException(status_code=400, detail="Invalid data format")

    try:
        # 异步生成图片
        image_data = await generate_image(data)
        image_size = len(image_data) / 1024 # 转换为KB
        logging.info(f"Image generated successfully - Size: {image_size:.2f} KB")

        # 保存图片文件
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"images/{timestamp}_{client_ip.replace('.', '_')}.png"
        with open(filename, 'wb') as f:
            f.write(image_data)

        # 转换为base64
        image_base64 = base64.b64encode(image_data).decode('utf-8')

        return JSONResponse({
            'image': image_base64,
            'message': 'Chart generated successfully',
            'saved_path': filename
        })
    except Exception as e:
        logging.error(f"Image generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=" 192.168.43.6", port=8000)
