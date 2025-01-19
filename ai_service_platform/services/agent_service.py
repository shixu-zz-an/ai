import json
import os
from typing import Dict, List, Optional, Generator, Any
import requests
from config import QWEN_API_URL, QWEN_MODEL

class AgentService:
    def __init__(self):
        self.agents_dir = 'data/agents'
        os.makedirs(self.agents_dir, exist_ok=True)
        self._load_agents()

    def _load_agents(self):
        """Load all agents from disk"""
        self.agents = {}
        for filename in os.listdir(self.agents_dir):
            if filename.endswith('.json'):
                with open(os.path.join(self.agents_dir, filename), 'r', encoding='utf-8') as f:
                    agent = json.load(f)
                    self.agents[agent['id']] = agent

    def _save_agent(self, agent: Dict):
        """Save agent to disk"""
        filename = f"{agent['id']}.json"
        with open(os.path.join(self.agents_dir, filename), 'w', encoding='utf-8') as f:
            json.dump(agent, f, ensure_ascii=False, indent=2)

    def create_agent(self, name: str, description: str, prompt: str, functions: List[str]) -> Dict:
        """Create a new agent"""
        agent_id = str(len(self.agents) + 1)  # 简单的 ID 生成
        agent = {
            'id': agent_id,
            'name': name,
            'description': description,
            'prompt': prompt,
            'functions': functions
        }
        self.agents[agent_id] = agent
        self._save_agent(agent)
        return agent

    def get_agent(self, agent_id: str) -> Optional[Dict]:
        """Get agent by ID"""
        return self.agents.get(agent_id)

    def update_agent(self, agent_id: str, name: str, description: str, prompt: str, functions: List[str]) -> Optional[Dict]:
        """Update an existing agent"""
        if agent_id not in self.agents:
            return None
        
        agent = {
            'id': agent_id,
            'name': name,
            'description': description,
            'prompt': prompt,
            'functions': functions
        }
        self.agents[agent_id] = agent
        self._save_agent(agent)
        return agent

    def delete_agent(self, agent_id: str) -> bool:
        """Delete an agent"""
        if agent_id not in self.agents:
            return False
        
        filename = f"{agent_id}.json"
        try:
            os.remove(os.path.join(self.agents_dir, filename))
            del self.agents[agent_id]
            return True
        except:
            return False

    def list_agents(self) -> List[Dict]:
        """List all agents"""
        return list(self.agents.values())

    def get_intent(self, message: str) -> str:
        """使用 Qwen 识别用户意图"""
        system_prompt = """你是一个意图识别专家。你需要分析用户的输入，并从以下选项中选择最匹配的意图：
            1. CHAT - 普通聊天对话
            2. QUERY - 文档检索查询
            3. AGENT - 需要特定 Agent 处理
            请只返回意图代码（CHAT/QUERY/AGENT），不要包含其他内容。"""

        try:
            response = requests.post(
                QWEN_API_URL,
                json={
                    "model": QWEN_MODEL,
                    "prompt": f"{system_prompt}\n\n用户输入: {message}\n\n意图:",
                    "stream": False
                }
            )
            response.raise_for_status()
            result = response.json()
            intent = result['response'].strip().upper()
            print(f"识别到的意图: {intent}")
            return intent if intent in ['CHAT', 'QUERY', 'AGENT'] else 'CHAT'
        except Exception as e:
            print(f"意图识别失败: {str(e)}")
            return "CHAT"  # 默认返回聊天意图

    def handle_rag_query(self, message: str) -> Generator[str, None, None]:
        """处理文档检索查询"""
        try:
            from services.rag_service import RAGService
            rag_service = RAGService()
            result = rag_service.query(message)
            
            # 首先发送文档来源
            yield json.dumps({
                'type': 'sources',
                'content': result['sources']
            })
            
            # 然后发送回答
            for chunk in result['answer_generator']:
                yield json.dumps({
                    'type': 'answer',
                    'content': chunk
                })
        except Exception as e:
            print(f"RAG 查询失败: {str(e)}")
            yield json.dumps({
                'type': 'error',
                'content': f"文档检索失败: {str(e)}"
            })

    def handle_agent_request(self, message: str, agent_id: str) -> Generator[str, None, None]:
        """处理 Agent 请求"""
        try:
            agent = self.get_agent(agent_id)
            if not agent:
                yield json.dumps({
                    'type': 'error',
                    'content': 'Agent not found'
                })
                return
            
            # TODO: 实现 Agent 特定的处理逻辑
            yield json.dumps({
                'type': 'message',
                'content': f"使用 {agent['name']} 处理消息: {message}"
            })
        except Exception as e:
            print(f"Agent 处理失败: {str(e)}")
            yield json.dumps({
                'type': 'error',
                'content': f"Agent 处理失败: {str(e)}"
            })

    def handle_chat(self, message: str) -> Generator[str, None, None]:
        """处理普通聊天"""
        try:
            print(f"使用 Qwen 处理消息: {message}")
            response = requests.post(
                QWEN_API_URL,
                json={
                    "model": QWEN_MODEL,
                    "prompt": message,
                    "stream": True
                },
                stream=True
            )
            response.raise_for_status()
            
            # 处理流式响应
            for line in response.iter_lines():
                if line:
                    try:
                        result = json.loads(line.decode())
                        if 'response' in result:
                            yield json.dumps({
                                'type': 'message',
                                'content': result['response']
                            })
                    except json.JSONDecodeError as e:
                        print(f"解析响应失败: {str(e)}")
                        continue
                        
        except Exception as e:
            print(f"聊天处理失败: {str(e)}")
            yield json.dumps({
                'type': 'error',
                'content': f"聊天处理失败: {str(e)}"
            })

    def process_message(self, message: str, agent_id: Optional[str] = None) -> Generator[str, None, None]:
        """处理用户消息"""
        try:
            # 识别意图
            intent = self.get_intent(message)
            print(f"处理消息，意图: {intent}")
            
            # 根据意图分发到不同的处理器
            if intent == "QUERY":
                yield from self.handle_rag_query(message)
            elif intent == "AGENT" and agent_id:
                yield from self.handle_agent_request(message, agent_id)
            else:
                yield from self.handle_chat(message)
                    
        except Exception as e:
            print(f"处理消息失败: {str(e)}")
            yield json.dumps({
                'type': 'error',
                'content': f"处理消息时出错: {str(e)}"
            })
