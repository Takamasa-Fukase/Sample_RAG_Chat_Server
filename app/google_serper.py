from typing import Any, List
from pydantic import BaseModel
from langchain.utilities import GoogleSerperAPIWrapper


class SerperResult(BaseModel):
    answer_box: str
    knowledge_graph: str
    organic_results_text: str
    links: List[str]

class CustomGoogleSerper(GoogleSerperAPIWrapper):
    k: int = 3
    gl: str = "jp"
    hl: str = "ja"

    def run(
        self,
        query: str, 
        **kwargs: Any
    ) -> SerperResult:
        results = super()._google_serper_api_results(
            query,
            gl=self.gl,
            hl=self.hl,
            num=self.k,
            **kwargs,
        )
        return self._parse_results(results=results)
    
    def _parse_results(
        self, 
        results: dict,
    ) -> SerperResult:
        print(f'⭐️_parse_snippets: {results}\n\n')
        answer_box_result: str = ""
        knowledge_graph_result: str = ""
        links: List[str] = []
        organic_results_text: str = ""

        # AnswerBoxの値が取れていたら整形して変数に格納する
        if (answer_box := results.get("answerBox")) is not None:
            print(f'answerBoxがある: {answer_box}')
            if (answer := answer_box.get("answer")) is not None:
                print(f'answer: {answer}')
                answer_box_result = answer
            elif (snippet := answer_box.get("snippet")) is not None:
                print(f'snippet: {snippet}')
                answer_box_result = snippet.replace("\n", " ")
            elif (highlighted_snippets := answer_box.get("snippetHighlighted")) is not None:
                print(f'snippet: {highlighted_snippets}')
                answer_box_result = '\n'.join(highlighted_snippets)

        # KnowledgeGraphの値が取れていたら整形して変数に格納する
        if (knowledge_graph := results.get("knowledgeGraph")) is not None:
            print(f'knowledgeGraphがある: {knowledge_graph}')
            title = knowledge_graph.get("title")
            entity_type = knowledge_graph.get("type")
            description = knowledge_graph.get("description")
            print(f'title: {title}, entity_type: {entity_type}, description: {description}')
            if entity_type:
                knowledge_graph_result += f"{title}: {entity_type}.\n"
            if description:
                print(f'description: {description}')
                knowledge_graph_result += f"{description}\n"
            for attribute, value in knowledge_graph.get("attributes", {}).items():
                print(f'attribute: {attribute}, value: {value}')
                knowledge_graph_result += f"{title} {attribute}: {value}.\n"

        for result in results[self.result_key_for_type[self.type]][: self.k]:
            # リンクを取り出してリストに格納
            if (link := result.get('link')) is not None:
                links.append(link)

            # 浅い情報だがsnippet部分も取り出してorganic_resultとして取得しておく（ディープサーチがOFFの場合はこれを使う）
            link = result.get("link", "")
            snippet = result.get("snippet", "")
            attributes = {f"{attribute}": value for attribute, value in result.get("attributes", {}).items()}
            organic_result = {"snippet": f"{snippet}. {attributes}", "link": link}
            organic_results_text += f"{organic_result}"

        return SerperResult(
            answer_box=answer_box_result,
            knowledge_graph=knowledge_graph_result,
            organic_results_text=organic_results_text,
            links=links,
        )