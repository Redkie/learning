from fastapi.responses import JSONResponse
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse, FileResponse
from starlette.staticfiles import StaticFiles

# 导入SPC控制图类
from SPC import (

    IControlChart,   # 1
    IMRControlChart,  # 2
    XSControlChart,  # 3
    XRControlChart,  # 4
    CControlChart,   # 5
    UControlChart,   # 6
    NPControlChart,  # 7
    PControlChart,  # 8
    Quan_SixChart   # 9

)

# 导入判异规则类
from SPC import (
    Rule01, Rule02, Rule03, Rule04,
    Rule05, Rule06, Rule07, Rule08
)

# 导入其他相关包
import base64
import logging
import logging.config
import os
from datetime import datetime
from typing import Dict, Any, Tuple, Optional
import aiofiles
from contextlib import asynccontextmanager

# 配置日志
LOG_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        },
    },
    'handlers': {
        'default': {
            'class': 'logging.StreamHandler',
            'formatter': 'standard',
        },
        **{
            chart_type: {
                'class': 'logging.FileHandler',
                'filename': f'logs/{chart_type}.log',
                'formatter': 'standard',
            } for chart_type in [
                'IChart', 'IMRChart', 'XSChart', 'XRChart',
                'CChart', 'SixPackChart', 'PChart', 'NPChart', 'UChart'
            ]
        }
    },
    'loggers': {
        f'api/generate_{chart_type}': {
            'handlers': [chart_type],
            'level': 'INFO',
            'propagate': False,
        } for chart_type in [
            'CChart', 'SixPackChart', 'IChart',
            'IMRChart', 'XSChart', 'XRChart', 'SixPackChart', 'PChart', 'NPChart', 'UChart'
        ]
    }
}


# 应用生命周期管理
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时创建目录
    os.makedirs("static/images", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    # 配置日志
    logging.config.dictConfig(LOG_CONFIG)
    for chart_type in [
        'CChart', 'UChart', 'PChart', 'NPChart',
        'SixPackChart', 'IChart', 'IMRChart', 'XSChart', 'XRChart'
    ]:
        os.makedirs(f'static/images/{chart_type}', exist_ok=True)
    yield


# 创建app
app = FastAPI(lifespan=lifespan)

# 配置从环境变量获取
LOCAL_IP = os.getenv('LOCAL_IP', '10.17.2.202')
PORT = int(os.getenv('PORT', '8884'))

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 静态文件服务
app.mount("/static", StaticFiles(directory="static"), name="static")


# 公共工具函数
def setup_logger(endpoint: str):
    return logging.getLogger(f'api/generate_{endpoint}')


async def validate_data(data: Dict[str, Any], required_fields: list = None) -> Tuple[list, list]:
    data_org = data.get('data', [])
    if not data_org or not all('edit_time' in item and 'data' in item for item in data_org):
        raise HTTPException(status_code=400, detail="Invalid data format")

    edit_time_list = [item['edit_time'] for item in data_org]
    data_list = [item['data'] for item in data_org]

    if required_fields:
        for field in required_fields:
            if field not in data:
                raise HTTPException(status_code=400, detail=f"Missing required field: {field}")

    return edit_time_list, data_list

# 获取带批数的方法
async def validate_group_data(data: Dict[str, Any], required_fields: list = None) -> Tuple[list, list, list]:
    data_org = data.get('data', [])
    if not data_org or not all('edit_time' in item and 'data' in item and 'group' in item for item in data_org):
        raise HTTPException(status_code=400, detail="Invalid data format")

    edit_time_list = [item['edit_time'] for item in data_org]
    data_list = [item['data'] for item in data_org]
    group_list = [item['group'] for item in data_org]

    if required_fields:
        for field in required_fields:
            if field not in data:
                raise HTTPException(status_code=400, detail=f"Missing required field: {field}")

    return edit_time_list, data_list, group_list



def apply_rules(chart, rule_list: list):
    rule_map = {
        1: Rule01, 2: Rule02, 3: Rule03, 4: Rule04,
        5: Rule05, 6: Rule06, 7: Rule07, 8: Rule08
    }
    for rule_id in set(rule_list):
        if rule_class := rule_map.get(rule_id):
            chart.append_rule(rule_class())


# 图表处理基类
class ChartHandler:
    def __init__(self, chart_type: str):
        self.chart_type = chart_type
        self.image_dir = f"static/images/{chart_type}"

    async def handle_request(self, request: Request, data: Dict[str, Any]):
        logger = setup_logger(self.chart_type)
        client_ip = request.client.host
        logger.info(f"IP: {client_ip} - {request.method} {request.url.path}")

        try:
            image_data, message, is_stable = await self.generate_chart(data)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{self.image_dir}/{timestamp}_{client_ip.replace('.', '_')}.png"

            async with aiofiles.open(filename, 'wb') as f:
                await f.write(image_data)

            image_base64 = base64.b64encode(image_data).decode('utf-8')
            view_url = f"{LOCAL_IP}:{PORT}/show-image?image_url=/static/images/{self.chart_type}/{timestamp}_{client_ip.replace('.', '_')}.png"

            return JSONResponse({
                'msg': 'Chart generated successfully',
                'code': 200,
                'view_picture': view_url,
                "message": message,
                "is_stable": is_stable,
                'data': image_base64
            })
        except HTTPException as he:
            raise he
        except Exception as e:
            logger.error(f"Image generation failed: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))


# 具体图表处理类
class SixPackHandler(ChartHandler):
    def __init__(self):
        super().__init__("SixPackChart")
11
    async def generate_chart(self, data: Dict[str, Any]):
        edit_time_list, data_list = await validate_data(data, ['group_size'])
        groupsize = data.get('group_size', 1)
        lower = data.get('lspec', None)
        upper = data.get('uspec', None)
        if not isinstance(groupsize, int) or groupsize < 2:
            raise HTTPException(status_code=400, detail="Invalid group_size, please check")
        if lower is None or  upper is None or not isinstance(upper,(int,float)) or not isinstance(lower,(int,float)):
            raise HTTPException(status_code=400, detail="invalid upper or lower, please check")
        chart = Quan_SixChart(
            data_list=data_list,
            edit_time_list=edit_time_list,
            groupSize=groupsize,
            lower=lower,
            upper=upper,
            title_name=data.get('chart_title', 'Six Pack Chart')
        )
        chart.dateformat = "%Y-%m-%d %H:%M:%S"
        apply_rules(chart, data.get('test', []))
        img, msg = chart.plot()
        return img.getvalue(), msg, chart.stable()


class CChartHandler(ChartHandler):
    def __init__(self):
        super().__init__("CChart")

    async def generate_chart(self, data: Dict[str, Any]):
        edit_time_list, data_list = await validate_data(data)

        chart = CControlChart(
            data_list=data_list,
            edit_time_list=edit_time_list,
            cchart_upper_line=data.get('c_uspec', None),
            cchart_lower_line=data.get('c_lspec', None),
            title_name=data.get('chart_title', 'C Control Chart')
        )
        chart.dateformat = "%Y-%m-%d %H:%M:%S"
        chart.limits = True
        apply_rules(chart, data.get('test', []))
        img, msg = chart.plot()
        return img.getvalue(), msg, chart.stable()

class UChartHandler(ChartHandler):
    def __init__(self):
        super().__init__("UChart")

    async def generate_chart(self, data: Dict[str, Any]):
        edit_time_list, data_list, group_list = await validate_group_data(data)

        chart = UControlChart(
            num_list=data_list,
            edit_time_list=edit_time_list,
            total_list=group_list,
            uchart_upper_line=data.get('u_uspec', None),
            uchart_lower_line=data.get('u_lspec', None),
            title_name=data.get('chart_title', 'U Control Chart')
        )
        chart.dateformat = "%Y-%m-%d %H:%M:%S"
        chart.limits = True
        apply_rules(chart, data.get('test', []))
        img, msg = chart.plot()
        return img.getvalue(), msg, chart.stable()


class NPChartHandler(ChartHandler):
    def __init__(self):
        super().__init__("NPChart")

    async def generate_chart(self, data: Dict[str, Any]):
        edit_time_list, data_list, group_list = await validate_group_data(data)

        chart = NPControlChart(
            num_list=data_list,
            edit_time_list=edit_time_list,
            total_list=group_list,
            npchart_upper_line=data.get('np_uspec', None),
            npchart_lower_line=data.get('np_lspec', None),
            title_name=data.get('chart_title', 'NP Control Chart')
        )
        chart.dateformat = "%Y-%m-%d %H:%M:%S"
        chart.limits = True
        apply_rules(chart, data.get('test', []))
        img, msg = chart.plot()
        return img.getvalue(), msg, chart.stable()


class PChartHandler(ChartHandler):
    def __init__(self):
        super().__init__("PChart")

    async def generate_chart(self, data: Dict[str, Any]):
        edit_time_list, data_list, group_list = await validate_group_data(data)

        chart = PControlChart(
            num_list=data_list,
            edit_time_list=edit_time_list,
            total_list= group_list,
            pchart_upper_line=data.get('p_uspec', None),
            pchart_lower_line=data.get('p_lspec', None),
            title_name=data.get('chart_title', 'P Control Chart')
        )
        chart.dateformat = "%Y-%m-%d %H:%M:%S"
        chart.limits = True
        apply_rules(chart, data.get('test', []))
        img, msg = chart.plot()
        return img.getvalue(), msg, chart.stable()

class IChartHandler(ChartHandler):
    def __init__(self):
        super().__init__("IChart")

    async def generate_chart(self, data: Dict[str, Any]):
        edit_time_list, data_list = await validate_data(data)

        chart = IControlChart(
            data_list=data_list,
            edit_time_list=edit_time_list,
            ichart_upper_line=data.get('i_uspec'),
            ichart_lower_line=data.get('i_lspec'),
            title_name=data.get('chart_title', 'I Control Chart')
        )
        chart.dateformat = "%Y-%m-%d %H:%M:%S"
        apply_rules(chart, data.get('test', []))
        img, msg = chart.plot()
        return img.getvalue(), msg, chart.stable()


class IMRChartHandler(ChartHandler):
    def __init__(self):
        super().__init__("IMRChart")

    async def generate_chart(self, data: Dict[str, Any]):
        edit_time_list, data_list = await validate_data(data)

        chart = IMRControlChart(
            data_list=data_list,
            edit_time_list=edit_time_list,
            i_upper_line=data.get('i_uspec'),
            i_lower_line=data.get('i_lspec'),
            mr_upper_line=data.get('mr_uspec'),
            mr_lower_line=data.get('mr_lspec'),
            title_name=data.get('chart_title', 'IMR Control Chart')
        )
        chart.dateformat = "%Y-%m-%d %H:%M:%S"
        apply_rules(chart, data.get('test', []))
        img, msg = chart.plot()
        return img.getvalue(), msg, chart.stable()


class XSChartHandler(ChartHandler):
    def __init__(self):
        super().__init__("XSChart")

    async def generate_chart(self, data: Dict[str, Any]):
        edit_time_list, data_list = await validate_data(data, ['group_size'])
        groupsize = data.get('group_size', 1)

        if not isinstance(groupsize, int) or groupsize < 2:
            raise HTTPException(status_code=400, detail="Invalid group_size")

        chart = XSControlChart(
            data_list=data_list,
            edit_time_list=edit_time_list,
            groupSize=groupsize,
            x_upper_line=data.get('x_uspec'),
            x_lower_line=data.get('x_lspec'),
            s_upper_line=data.get('s_uspec'),
            s_lower_line=data.get('s_lspec'),
            title_name=data.get('chart_title', 'X-S Control Chart')
        )
        chart.dateformat = "%Y-%m-%d %H:%M:%S"
        apply_rules(chart, data.get('test', []))
        img, msg = chart.plot()
        return img.getvalue(), msg, chart.stable()


class XRChartHandler(ChartHandler):
    def __init__(self):
        super().__init__("XRChart")

    async def generate_chart(self, data: Dict[str, Any]):
        edit_time_list, data_list = await validate_data(data, ['group_size'])
        groupsize = data.get('group_size', 2)

        if not isinstance(groupsize, int) or groupsize < 2:
            raise HTTPException(status_code=400, detail="Invalid group_size")

        chart = XRControlChart(
            data_list=data_list,
            edit_time_list=edit_time_list,
            groupSize=groupsize,
            x_upper_line=data.get('x_uspec'),
            x_lower_line=data.get('x_lspec'),
            r_upper_line=data.get('r_uspec'),
            r_lower_line=data.get('r_lspec'),
            title_name=data.get('chart_title', 'X-R Control Chart')
        )
        chart.dateformat = "%Y-%m-%d %H:%M:%S"
        apply_rules(chart, data.get('test', []))
        img, msg = chart.plot()
        return img.getvalue(), msg, chart.stable()


# 路由处理器注册
handlers = {
    "SixPackChart": SixPackHandler(),
    "CChart": CChartHandler(),
    "IChart": IChartHandler(),
    "IMRChart": IMRChartHandler(),
    "XSChart": XSChartHandler(),
    "XRChart": XRChartHandler(),
    "PChart": PChartHandler(),
    "NPChart": NPChartHandler(),
    "UChart": UChartHandler()
}


@app.post("/api/generate_{chart_type}")
async def generate_chart(chart_type: str, request: Request):
    handler = handlers.get(chart_type)
    if not handler:
        raise HTTPException(status_code=404, detail="Chart type not found")

    try:
        data = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON data")

    return await handler.handle_request(request, data)


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.get('/favicon.ico', include_in_schema=False)
async def favicon():
    return FileResponse('static/fav.ico')


@app.get("/show-image")
async def show_image(image_url: str):
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Image Viewer</title>
        <style>
            body {{
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
                margin: 0;
            }}
            img {{
                max-width: 100%;
                max-height: 100%;
            }}
        </style>
    </head>
    <body>
        <img src="{image_url}" alt="Displayed Image">
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=LOCAL_IP, port=PORT,reload=False)