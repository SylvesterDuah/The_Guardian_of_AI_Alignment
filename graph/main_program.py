# %%
# Import Libraries
import os
import re
import numpy as np
from math import ceil
import matplotlib.pyplot as plt
import pandas as pd

import networkx as nx
import requests
import tweepy
import scipy
from uuid import uuid4
from collections import Counter


import smtplib
from email.mime.text import MIMEText
import os
from playsound3 import playsound 


from dotenv import load_dotenv

from langchain.agents import tool, initialize_agent
from arango import ArangoClient
from langchain_groq import ChatGroq 
from langchain.prompts import ChatPromptTemplate
from langchain_community.graphs.arangodb_graph import ArangoGraph
from langchain.chains import ArangoGraphQAChain


# %%
from langchain_community.graphs import ArangoGraph
from arango import ArangoClient
from arango.exceptions import ArangoClientError, ArangoServerError
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_arangodb_connection(db_name="graph_db", graph_name="ai_incidents_graph"):
    """
    Set up the ArangoDB connection and create the graph and collections if they don't exist.
    """
    try:
        # Initialize ArangoDB client
        client = ArangoClient(hosts="http://localhost:8529")  # Replace with your host details if needed

        # Connect to the _system database for administrative tasks
        sys_db = client.db("_system", username="root", password="newpassword")
        
        # Check if the database exists, and create it if it doesn't
        if not sys_db.has_database(db_name):
            sys_db.create_database(db_name)
            logger.info("Created new database: %s", db_name)
        
        # Connect to the target database
        db = client.db(db_name, username="root", password="newpassword")
        logger.info("Successfully connected to ArangoDB database: %s", db_name)

        # Create the graph if it doesn't exist
        if not db.has_graph(graph_name):
            graph = db.create_graph(graph_name)
            logger.info("Created new graph: %s", graph_name)
        else:
            graph = db.graph(graph_name)
            logger.info("Using existing graph: %s", graph_name)

        # Create collections for nodes and edges if they don't exist
        nodes_collection_name = "nodes"
        edges_collection_name = "edges"

        if not graph.has_vertex_collection(nodes_collection_name):
            nodes_collection = graph.create_vertex_collection(nodes_collection_name)
            logger.info("Created vertex collection: %s", nodes_collection_name)
        else:
            nodes_collection = graph.vertex_collection(nodes_collection_name)
            logger.info("Using existing vertex collection: %s", nodes_collection_name)

        if not graph.has_edge_definition(edges_collection_name):
            edges_collection = graph.create_edge_definition(
                edge_collection=edges_collection_name,
                from_vertex_collections=[nodes_collection_name],
                to_vertex_collections=[nodes_collection_name]
            )
            logger.info("Created edge collection: %s", edges_collection_name)
        else:
            edges_collection = graph.edge_collection(edges_collection_name)
            logger.info("Using existing edge collection: %s", edges_collection_name)

        return db, graph, nodes_collection, edges_collection

    except ArangoClientError as e:
        logger.error("Failed to connect to ArangoDB: %s", str(e))
        raise
    except ArangoServerError as e:
        logger.error("ArangoDB server error: %s", str(e))
        raise
    except Exception as e:
        logger.error("Unexpected error: %s", str(e))
        raise

def create_arangograph_wrapper(db):
    """
    Create the ArangoGraph LangChain wrapper.
    """
    try:
        arango_graph = ArangoGraph(db)
        logger.info("Created ArangoGraph wrapper successfully.")
        return arango_graph
    except Exception as e:
        logger.error("Failed to create ArangoGraph wrapper: %s", str(e))
        raise

def main():
    try:
        # Step 1: Set up ArangoDB connection
        db, graph, nodes_collection, edges_collection = setup_arangodb_connection()

        # Step 2: Create the ArangoGraph LangChain wrapper
        arango_graph = create_arangograph_wrapper(db)

        # Step 3: Use the wrapper to query the graph
        print("ArangoGraph wrapper created successfully.")

        # Query the number of nodes and edges
        num_nodes = arango_graph.query("RETURN LENGTH(nodes)")
        num_edges = arango_graph.query("RETURN LENGTH(edges)")

        print("Number of nodes:", num_nodes[0] if num_nodes else "Unknown")
        print("Number of edges:", num_edges[0] if num_edges else "Unknown")

    except Exception as e:
        logger.error("Error in main workflow: %s", str(e))

# Run the main workflow
if __name__ == "__main__":
    main()

# %%
from langchain_community.graphs import ArangoGraph
from arango import ArangoClient
from arango.exceptions import ArangoClientError, ArangoServerError
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_arangodb_connection(db_name="graph_db", graph_name="ai_incidents_graph"):
    """
    Set up the ArangoDB connection and create the graph and collections if they don't exist.
    """
    try:
        # Initialize ArangoDB client
        client = ArangoClient(hosts="http://localhost:8529")  # Replace with your host details if needed

        # Connect to the _system database for administrative tasks
        sys_db = client.db("_system", username="root", password="newpassword")
        
        # Check if the database exists, and create it if it doesn't
        if not sys_db.has_database(db_name):
            sys_db.create_database(db_name)
            logger.info("Created new database: %s", db_name)
        
        # Connect to the target database
        db = client.db(db_name, username="root", password="newpassword")
        logger.info("Successfully connected to ArangoDB database: %s", db_name)

        # Create the graph if it doesn't exist
        if not db.has_graph(graph_name):
            graph = db.create_graph(graph_name)
            logger.info("Created new graph: %s", graph_name)
        else:
            graph = db.graph(graph_name)
            logger.info("Using existing graph: %s", graph_name)

        # Create collections for nodes and edges if they don't exist
        nodes_collection_name = "nodes"
        edges_collection_name = "edges"

        if not graph.has_vertex_collection(nodes_collection_name):
            nodes_collection = graph.create_vertex_collection(nodes_collection_name)
            logger.info("Created vertex collection: %s", nodes_collection_name)
        else:
            nodes_collection = graph.vertex_collection(nodes_collection_name)
            logger.info("Using existing vertex collection: %s", nodes_collection_name)

        if not graph.has_edge_definition(edges_collection_name):
            edges_collection = graph.create_edge_definition(
                edge_collection=edges_collection_name,
                from_vertex_collections=[nodes_collection_name],
                to_vertex_collections=[nodes_collection_name]
            )
            logger.info("Created edge collection: %s", edges_collection_name)
        else:
            edges_collection = graph.edge_collection(edges_collection_name)
            logger.info("Using existing edge collection: %s", edges_collection_name)

        return db, graph, nodes_collection, edges_collection

    except ArangoClientError as e:
        logger.error("Failed to connect to ArangoDB: %s", str(e))
        raise
    except ArangoServerError as e:
        logger.error("ArangoDB server error: %s", str(e))
        raise
    except Exception as e:
        logger.error("Unexpected error: %s", str(e))
        raise

def create_arangograph_wrapper(db):
    """
    Create the ArangoGraph LangChain wrapper.
    """
    try:
        arango_graph = ArangoGraph(db)
        logger.info("Created ArangoGraph wrapper successfully.")
        return arango_graph
    except Exception as e:
        logger.error("Failed to create ArangoGraph wrapper: %s", str(e))
        raise

def main():
    try:
        # Step 1: Set up ArangoDB connection
        db, graph, nodes_collection, edges_collection = setup_arangodb_connection()

        # Step 2: Create the ArangoGraph LangChain wrapper
        arango_graph = create_arangograph_wrapper(db)

        # Step 3: Use the wrapper to query the graph
        print("ArangoGraph wrapper created successfully.")

        # Query the number of nodes and edges
        num_nodes = arango_graph.query("RETURN LENGTH(nodes)")
        num_edges = arango_graph.query("RETURN LENGTH(edges)")

        print("Number of nodes:", num_nodes[0] if num_nodes else "Unknown")
        print("Number of edges:", num_edges[0] if num_edges else "Unknown")

    except Exception as e:
        logger.error("Error in main workflow: %s", str(e))

# Run the main workflow
if __name__ == "__main__":
    main()

# %%
def load_graph_from_processed_data(processed_data_folder):
    """
    Load processed data (nodes and edges) into a NetworkX graph.
    """
    G = nx.Graph()

    # Load nodes
    for file in os.listdir(processed_data_folder):
        if file.endswith("_nodes.csv"):
            nodes_file = os.path.join(processed_data_folder, file)
            try:
                nodes_df = pd.read_csv(nodes_file)
                for _, row in nodes_df.iterrows():
                    node_id = row.get("node_id") 
                    if node_id:
                        G.add_node(node_id, **row.to_dict())
            except Exception as e:
                print(f"Error processing node file '{file}': {e}")

    # Load edges
    for file in os.listdir(processed_data_folder):
        if file.endswith("_edges.csv"):
            edges_file = os.path.join(processed_data_folder, file)
            try:
                edges_df = pd.read_csv(edges_file)

                # Matching column pairs
                column_pairs = [
                    ("authors", "source_domain"),
                    ("Alleged deployer of AI system", "Alleged developer of AI system"),
                    ("Known AI Goal", "Known AI Technology"),
                    ("duplicate_incident_number", "true_incident_number")
                ]
                
                source_col, target_col = None, None
                for col1, col2 in column_pairs:
                    if col1 in edges_df.columns and col2 in edges_df.columns:
                        source_col, target_col = col1, col2
                        break
                
                if not source_col or not target_col:
                    print(f"Skipping file '{file}': No matching column structure.")
                    continue

                seen_edges = set()
                for _, row in edges_df.iterrows():
                    source = row[source_col]
                    target = row[target_col]
                    weight = row.get("weight", 1)
                    if (source, target) not in seen_edges and (target, source) not in seen_edges:
                        G.add_edge(source, target, weight=weight)
                        seen_edges.add((source, target))
            except Exception as e:
                print(f"Error processing edge file '{file}': {e}")
                continue

    return G


# %%
# Path to the processed data folder
processed_data_folder = "/Users/sylvesterduah/Documents/Code/De_Alignment/data/processed/"

# Load the graph from processed data
G = load_graph_from_processed_data(processed_data_folder)

# Print basic graph information
print("Graph Info:")
print(f"Number of nodes: {G.number_of_nodes()}")
print(f"Number of edges: {G.number_of_edges()}")
print(f"Nodes (first 5): {list(G.nodes)[:5]}")
print(f"Edges (first 5): {list(G.edges)[:5]}")

# %%
plot_options = {"node_size": 10, "with_labels": False, "width": 0.15}
pos = nx.spring_layout(G, iterations=15, seed=1721)
fig, ax = plt.subplots(figsize=(15, 9))
nx.draw_networkx(G, pos=pos, ax=ax, **plot_options)
plt.title("Graph Visualization")
plt.show()

# Save the graph in GraphML format
nx.write_graphml(G, "graph.graphml")
print("Graph saved to 'graph.graphml'.")

# %%
print("Validating graph...")

if G.number_of_nodes() == 0:
    print("Error: No nodes found in the graph.")
if G.number_of_edges() == 0:
    print("Error: No edges found in the graph.")

isolated_nodes = list(nx.isolates(G))
if isolated_nodes:
    print(f"Warning: {len(isolated_nodes)} isolated nodes found.")

self_loops = list(nx.selfloop_edges(G))
if self_loops:
    print(f"Warning: {len(self_loops)} self-loops found.")

# %%
# Initialize the ArangoDB client
client = ArangoClient(hosts="http://localhost:8529")

# Connect to the _system database for admin tasks
sys_db = client.db("_system", username="root", password="newpassword")

# Create the database "graph_db" if it doesn't exist
if not sys_db.has_database("graph_db"):
    sys_db.create_database("graph_db")
    print("Database 'graph_db' created successfully")

# Connect to the newly created (or existing) database
db = client.db("graph_db", username="root", password="newpassword")

# Create or get the graph "ai_incidents_graph"
if not db.has_graph("ai_incidents_graph"):
    graph = db.create_graph("ai_incidents_graph")
else:
    graph = db.graph("ai_incidents_graph")

# Create vertex collection "nodes"
if not graph.has_vertex_collection("nodes"):
    nodes_collection = graph.create_vertex_collection("nodes")
else:
    nodes_collection = graph.vertex_collection("nodes")

# Create edge definition "edges"
if not graph.has_edge_definition("edges"):
    edges_collection = graph.create_edge_definition(
        edge_collection="edges",
        from_vertex_collections=["nodes"],
        to_vertex_collections=["nodes"]
    )
else:
    edges_collection = graph.edge_collection("edges")

print("ArangoDB setup completed successfully!")

# %%
def sanitize_key(key):
    """
    Sanitize a key to make it valid for ArangoDB.
    """
    sanitized_key = re.sub(r"[^a-zA-Z0-9_\-:]", "_", str(key))
    if not sanitized_key:
        sanitized_key = "default_key"
    return sanitized_key[:254]

def convert_numpy_types(obj):
    """
    Recursively convert numpy numeric types to native Python types.
    """
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(elem) for elem in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    else:
        return obj

def persist_graph_to_arangodb(G, nodes_collection, edges_collection):
    """
    Persist a NetworkX graph to ArangoDB.
    """
    key_mapping = {}
    
    # Insert nodes into ArangoDB
    for node, data in G.nodes(data=True):
        sanitized_key = sanitize_key(node)
        if sanitized_key in key_mapping:
            sanitized_key = f"{sanitized_key}_{uuid4().hex[:8]}"
        key_mapping[node] = sanitized_key
        
        # Convert data values to native Python types
        data_converted = convert_numpy_types(data)
        
        if not nodes_collection.has(sanitized_key):
            nodes_collection.insert({"_key": sanitized_key, **data_converted})
        else:
            print(f"Node '{sanitized_key}' already exists. Skipping insertion.")
    
    # Insert edges into ArangoDB
    for source, target, data in G.edges(data=True):
        sanitized_source = key_mapping[source]
        sanitized_target = key_mapping[target]
        
        # Convert edge data to native Python types
        data_converted = convert_numpy_types(data)
        
        edges_collection.insert({
            "_from": f"nodes/{sanitized_source}",
            "_to": f"nodes/{sanitized_target}",
            **data_converted
        })

def validate_graph_in_arangodb(nodes_collection, edges_collection):
    """
    Validate the graph in ArangoDB by checking node and edge counts.
    """
    node_count = nodes_collection.count()
    print(f"Number of nodes in ArangoDB: {node_count}")

    edge_count = edges_collection.count()
    print(f"Number of edges in ArangoDB: {edge_count}")

    sample_node = next(nodes_collection.all(), None)
    if sample_node:
        print(f"Sample node '{sample_node['_key']}': {sample_node}")

    sample_edge = next(edges_collection.all(), None)
    if sample_edge:
        print(f"Sample edge: {sample_edge}")

def main_persist():
    # Assume 'G', 'nodes_collection', and 'edges_collection' are defined globally.
    persist_graph_to_arangodb(G, nodes_collection, edges_collection)
    print("Graph persisted to ArangoDB.")
    validate_graph_in_arangodb(nodes_collection, edges_collection)

# Run the persistence workflow
main_persist()

# %%
def create_arangodb_backed_graph(db, graph_name="ai_incidents_graph"):
    """
    Create an ArangoDB-backed NetworkX graph.
    """
    if not db.has_graph(graph_name):
        raise ValueError(f"Graph '{graph_name}' does not exist in ArangoDB.")
    
    graph = db.graph(graph_name)
    nodes_collection = graph.vertex_collection("nodes")
    edges_collection = graph.edge_collection("edges")
    
    G_arango = nx.Graph()
    
    # Add nodes from ArangoDB
    for node in nodes_collection.all():
        G_arango.add_node(node["_key"], **node)
    
    # Add edges from ArangoDB
    for edge in edges_collection.all():
        source = edge["_from"].split("/")[1]
        target = edge["_to"].split("/")[1]
        G_arango.add_edge(source, target, **edge)
    
    return G_arango

G_from_arango = create_arangodb_backed_graph(db, graph_name="ai_incidents_graph")
print("ArangoDB-backed NetworkX Graph:")
print(f"Number of nodes: {G_from_arango.number_of_nodes()}")
print(f"Number of edges: {G_from_arango.number_of_edges()}")

# %%
def query_graph_in_arangodb(db, graph_name="ai_incidents_graph"):
    """
    Execute various AQL queries to interact with the graph.
    """
    graph = db.graph(graph_name)
    
    result = db.aql.execute("""
        FOR node IN nodes
            SORT RAND()
            LIMIT 3
            RETURN node
    """)
    print("Sample 3 nodes:")
    print(list(result))
    print('-' * 10)

    result = db.aql.execute("""
        FOR edge IN edges
            SORT RAND()
            LIMIT 3
            RETURN edge
    """)
    print("Sample 3 edges:")
    print(list(result))
    print('-' * 10)

    node_key = "Content_Recommendation__Content_Search__Hate_Speech_Detection__NSFW_Content_Detection_0"
    result = db.aql.execute(f"""
        FOR v, e, p IN 1..1 ANY 'nodes/{node_key}' GRAPH ai_incidents_graph
            LIMIT 1
            RETURN p
    """)
    print(f"1-hop neighborhood of node '{node_key}':")
    print(list(result))
    print('-' * 10)

    result = db.aql.execute("""
        FOR node IN nodes
            FILTER node.type == "AI System"
            LIMIT 3
            RETURN node
    """)
    print("Nodes with type 'AI System':")
    print(list(result))
    print('-' * 10)

    result = db.aql.execute("""
        FOR edge IN edges
            FILTER edge.weight > 1
            LIMIT 3
            RETURN edge
    """)
    print("Edges with weight > 1:")
    print(list(result))
    print('-' * 10)

query_graph_in_arangodb(db)

# %%
# Re-load the processed NetworkX graph and print info
G = load_graph_from_processed_data(processed_data_folder)
print("Graph Info (re-loaded):")
print(f"Number of nodes: {G.number_of_nodes()}")
print(f"Number of edges: {G.number_of_edges()}")
print(f"Nodes (first 5): {list(G.nodes)[:5]}")
print(f"Edges (first 5): {list(G.edges)[:5]}")

# %%
load_dotenv()

# Instantiate the ChatGroq client.
client = ChatGroq(
    model="llama3-70b-8192",  
    api_key=os.environ.get("GROQ_API_KEY"),
    temperature=0.5
)

def get_ai_response(prompt: str) -> str:
    """Get AI response safely with error handling."""
    try:
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"API Error: {str(e)}"

if __name__ == "__main__":
    prompt = "Explain how we can secure our planet by preventing AI misalignment"
    print(get_ai_response(prompt))

# %%
@tool
def misalignment(query: str) -> str:
    """A dummy tool that always returns a fixed guardian message."""
    return "I am the guardian!"

tools = [misalignment]

# %%
@tool
def text_to_aql_to_text(query: str):
    """
    Translate natural language to AQL using a direct DB connection.
    This version uses a highly refined prompt to enforce a single, flat AQL query.
    It explicitly instructs not to use extra nesting, UNION, INTO, or extra parentheses.
    If generation fails after a few attempts, it falls back gracefully.
    """

    # Wrap the native db connection with ArangoGraph
    wrapped_graph = ArangoGraph(db)


    max_attempts = 2

    llm = ChatGroq(
        model_name="llama3-70b-8192",
        api_key=os.environ.get("GROQ_API_KEY")
    )

    # Refined AQL
    aql_generation_prompt = (
        "You are an expert in ArangoDB AQL. The graph has the following collections:\n"
        "  - 'ai_nodes': Contains node documents with a unique '_key' and other attributes.\n"
        "  - 'edges': Contains edge documents with '_from' and '_to' fields referencing 'ai_nodes'.\n\n"
        "Your task is to generate a single, flat, syntactically correct AQL query based solely on the above collections.\n"
        "IMPORTANT: Do NOT use any nested subqueries, extra parentheses, UNION, INTO, or any keyword that is not required.\n"
        "For example, a valid query is:\n\n"
        "   FOR node IN ai_nodes\n"
        "       FILTER node._key == \"example_key\"\n"
        "       RETURN node\n\n"
        "Based on the following user input, produce a valid AQL query that adheres to these instructions.\n"
        "User Input: {user_input}\n"
        "AQL Query:"
    )

    chain = ArangoGraphQAChain.from_llm(
        llm=llm,
        graph=wrapped_graph,
        verbose=True,
        max_aql_generation_attempts=max_attempts,
        aql_generation_prompt=aql_generation_prompt
    )

    try:
        result = chain.invoke({"query": query})
    except ValueError as e:
        print("AQL generation failed after maximum attempts. Error details:")
        print(e)
        return "Fallback: Unable to generate a valid AQL query."

    if hasattr(chain, "last_generated_aql"):
        print("Generated AQL Query:", chain.last_generated_aql)

    return str(result["result"])

# %%
G.degree(107)

# %%
@tool
def text_to_nx_algorithm_to_text(query: str):
    """Execute NetworkX algorithm code on the graph and return results."""
    global client
    print("1) Generating NetworkX code")
    response = client.chat.completions.create(
        messages=[{
            "role": "user",
            "content": f"Generate NetworkX code for: {query}\nReturn ONLY valid Python code between ```python``` blocks"
        }],
        model="llama3-70b-8192"
    )
    code_text = response.choices[0].message.content
    cleaned_code = re.sub(r"^```python\n|```$", "", code_text, flags=re.MULTILINE).strip()
    print('-'*10)
    print("Generated code:\n", cleaned_code)
    print('-'*10)
    print("\n2) Executing NetworkX code")
    global_vars = {"G": G, "nx": nx}
    local_vars = {}
    try:
        exec(cleaned_code, global_vars, local_vars)
    except Exception as e:
        print(f"Execution error: {e}")
        return f"Error executing code: {e}"
    FINAL_RESULT = local_vars.get("FINAL_RESULT", "No result found")
    print("3) Formulating final answer")
    response2 = client.chat.completions.create(
        messages=[{
            "role": "user",
            "content": f"I analyzed a NetworkX Graph with AI incident data.\nExecuted code: {cleaned_code}\nResult: {FINAL_RESULT}\nAnswer this query concisely: {query}"
        }],
        model="llama3-70b-8192"
    )
    return response2.choices[0].message.content

# %%
def query_graph(query: str) -> str:
    """
    Query our graph using the agent.
    Fallback gracefully if the agent fails.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an intelligent agent that analyzes graph data and answers questions based on it."),
        ("human", "{input}")
    ])

    llm = ChatGroq(
        model_name="llama3-70b-8192",
        api_key=os.environ.get("GROQ_API_KEY")
    )

    agent = initialize_agent(
        tools,
        llm,
        agent="zero-shot-react-description",
        verbose=True
    )

    try:
        response = agent.run(query)
        return response
    except Exception as e:
        print("Query generation error encountered:", e)
        return "Fallback: Query skipped due to error."

# %%
# Refined queries tailored to the graph schema
queries = [
    "Retrieve the ai_node with _key 'Node0' and list its attributes.",
    "Find the shortest path between ai_nodes with _key 'Node0' and 'Node1'.",
    "Show all nodes directly connected to the ai_node with _key 'Node0'.",
    "What is the average degree of nodes in the graph?",
    "Identify nodes whose removal would significantly fragment the graph.",
    "Fetch the ai_node with the highest pagerank value.",
    "Which ai_node represents the most influential company in the graph?",
    "Retrieve all ai_nodes where the type attribute is 'AI System'.",
    "Find the ai_node with the highest connectivity.",
    "Determine how strongly connected the network is using connected components analysis.",
    "Identify the ai_node with the highest centrality value."
]

# Loop through the refined queries and print the responses
for q in queries:
    print("Query:", q)
    print("Response:", query_graph(q))
    print("=" * 50)

# %%
# Refined query for the project:
refined_query = (
    "Find the ai_node from the 'ai_nodes' collection representing a person, AI, institution, or company that has the highest connectivity. "
    "Determine popularity based on its degree (or pagerank if available) and explain why this entity is the most popular."
)

result = query_graph(refined_query)
print("Final Output:")
print(result)

# %%
twitter_token = os.environ.get("TWITTER_BEARER_TOKEN")

# %%

def query_graph(query: str) -> str:
    """
    Query our graph using the agent.
    If the agent fails to generate a valid query, it will fallback gracefully.
    """
    llm = ChatGroq(
        model_name="llama3-70b-8192",
        api_key=os.environ.get("GROQ_API_KEY")
    )
    agent = initialize_agent(
        tools,
        llm,
        agent="zero-shot-react-description",
        verbose=True
    )
    try:
        return agent.run(query)
    except Exception as e:
        print("Graph query error:", e)
        return ""


def fetch_tweets(query: str) -> list:
    """
    Fetch tweets from X (Twitter) using Tweepy and the Twitter API v2.
    Returns a list of dictionaries with keys: 'title', 'description', 'url'.
    """
    bearer_token = os.environ.get("TWITTER_BEARER_TOKEN")
    if not bearer_token:
        print("Error: TWITTER_BEARER_TOKEN is not set in the environment.")
        return []
    
    client = tweepy.Client(bearer_token=bearer_token)
    try:
        response = client.search_recent_tweets(query=query, tweet_fields=["id", "text"], max_results=10)
    except Exception as ex:
        print("Error fetching tweets:", ex)
        return []
    
    tweets = response.data
    if not tweets:
        return []
    
    results = []
    for tweet in tweets:
        tweet_id = tweet.id
        tweet_text = tweet.text
        tweet_url = f"https://twitter.com/i/web/status/{tweet_id}"
        results.append({
            "title": tweet_text[:50] + "..." if len(tweet_text) > 50 else tweet_text,
            "description": tweet_text,
            "url": tweet_url
        })
    return results

def compare_tweets_with_graph(query: str) -> str:
    """
    For a given query, fetch tweets from Twitter (X) and query our graph.
    Combine the outputs and return them.
    """
    tweets = fetch_tweets(query)
    graph_result = query_graph(query)
    
    def is_similar(text1: str, text2: str) -> bool:
        return text1.lower() in text2.lower() or text2.lower() in text1.lower()
    
    combined_output = ""
    if graph_result and tweets:
        similar_found = any(is_similar(tweet["title"], graph_result) for tweet in tweets)
        if similar_found:
            combined_output += "Combined Result:\nGraph Data:\n" + graph_result + "\n\nTweets:\n"
            for tweet in tweets:
                combined_output += f"Title: {tweet['title']}\nURL: {tweet['url']}\n\n"
        else:
            combined_output += "Both sources have data but no strong similarity was found.\n\nGraph Data:\n" + graph_result + "\n\nTweets:\n"
            for tweet in tweets:
                combined_output += f"Title: {tweet['title']}\nURL: {tweet['url']}\n\n"
    elif graph_result:
        combined_output = "Graph Data:\n" + graph_result
    elif tweets:
        combined_output = "Tweets:\n" + "\n".join([f"Title: {tweet['title']} - URL: {tweet['url']}" for tweet in tweets])
    else:
        combined_output = "Fallback: No relevant information found in either source."
    
    return combined_output


def analyze_misalignments(sector: str) -> str:
    """
    Analyze AI misalignments in a given sector by retrieving reports from various sources,
    then using the advanced agent to analyze and provide improvement suggestions with ethical references.
    """
    sample_reports = {
        "healthcare": (
            "Report: An AI diagnostic system in healthcare misaligned its treatment recommendations. "
            "Errors included misdiagnosis and failure to consider patient history. Reference: Journal of Medical AI, 2024."
        ),
        "autonomous vehicles": (
            "Report: Autonomous vehicle AI systems exhibited unsafe sensor fusion and decision-making errors. "
            "Reference: Autonomous Systems Quarterly, 2023."
        ),
        "cyber security": (
            "Report: AI-powered threat detection systems had high false positive rates and missed critical intrusions. "
            "Reference: Cyber Defense Review, 2023."
        ),
        "financial institutions": (
            "Report: Fraud detection algorithms in banks were found biased and inconsistent. "
            "Reference: Financial AI Insights, 2024."
        ),
        "ai companies": (
            "Report: Major AI companies like OpenAI, Microsoft, and Google faced model misalignment and governance issues. "
            "Reference: AI Ethics Journal, 2024."
        ),
        "individual projects": (
            "Report: Deployed AI projects online showed misalignment and unpredictable outputs. "
            "Reference: Online AI Deployment Forum, 2023."
        )
    }
    reports = sample_reports.get(sector.lower(), "")
    if not reports:
        return f"Fallback: No reports found for sector '{sector}'."
    
    analysis_query = (
        f"Analyze the following reports regarding AI misalignments in {sector}:\n\n"
        f"{reports}\n\n"
        "Identify the main faults in these AI systems and provide concrete improvement suggestions "
        "with references to AI ethics and governance best practices. If possible, compare these findings with "
        "similar issues reported in tweets. Provide a concise analysis."
    )
    
    try:
        analysis_result = query_graph(analysis_query)
    except Exception as e:
        print(f"Error analyzing reports for sector '{sector}':", e)
        return f"Fallback: Unable to analyze reports for sector '{sector}'."
    
    tweets_info = compare_tweets_with_graph(analysis_query)
    
    return f"Analysis Result:\n{analysis_result}\n\nAdditional Tweet Info:\n{tweets_info}"


# %%
def monitor_ai_misalignments():
    """
    Monitor AI activities across multiple sectors. For each sector, retrieve and analyze reports 
    from our graph and Reddit sources. If potential AI misalignment issues are detected, raise an alert.
    Falls back gracefully if no data is found or an error occurs.
    """
    # List of sectors to monitor
    sectors = [
        "healthcare",
        "autonomous vehicles",
        "cyber security",
        "financial institutions",
        "ai companies",
        "individual projects"
    ]
    
    alerts = []
    
    for sector in sectors:
        print(f"--- Monitoring sector: {sector} ---")
        try:
            analysis = analyze_misalignments(sector)
        except Exception as e:
            print(f"Error analyzing sector '{sector}': {e}")
            analysis = f"Fallback: Analysis error for sector '{sector}'."
        
        # If the analysis returns a fallback message, skip further processing for this sector
        if analysis.startswith("Fallback"):
            print(f"No sufficient data for sector '{sector}'. Skipping.")
            continue
        
        # Use a simple keyword check to determine if misalignment issues are flagged.
        keywords = ["misalign", "error", "fault", "unsafe", "inconsistent", "issue"]
        if any(keyword in analysis.lower() for keyword in keywords):
            alerts.append((sector, analysis))
            # Simulate an alarm alert
            print(f"ALERT: Potential AI misalignment detected in sector '{sector}'.")
        else:
            print(f"No misalignment issues detected in sector '{sector}'.")
    
    if alerts:
        radar_report = "Radar Report: Potential AI misalignments detected in the following sectors:\n"
        for sector, report in alerts:
            radar_report += f"\nSector: {sector}\nReport:\n{report}\n"
        return radar_report
    else:
        return "Radar Report: No significant AI misalignment issues detected across monitored sectors."

# Run the radar monitoring system
print(monitor_ai_misalignments())

# %%
user_accounts = {}

def query_graph(query: str) -> str:
    """
    Dummy implementation for querying the graph.
    Replace this with your actual graph querying function.
    """
    return f"Graph query result for: {query}"

def fetch_tweets(query: str) -> list:
    """
    Dummy implementation for fetching tweets.
    Replace this with your actual Tweepy-based tweet fetching function.
    """
    return []

def check_project_history(project: dict) -> str:
    """
    Check if the project's AI has previously misaligned by searching our graph and tweets.
    Returns a historical report if incidents are found; otherwise, a fallback message.
    """
    search_query = f"misalign {project['industry']} {project['description']}"
    graph_history = query_graph(search_query)
    tweets_history = fetch_tweets(search_query)
    
    if not graph_history and not tweets_history:
        return "No historical misalignment incidents found for this project."
    
    report = "Historical Misalignment Report:\n"
    if graph_history:
        report += f"Graph History:\n{graph_history}\n\n"
    if tweets_history:
        report += "Tweet History:\n"
        for tweet in tweets_history:
            report += f"Title: {tweet['title']}\nURL: {tweet['url']}\n\n"
    return report

def register_user(username: str, password: str) -> str:
    """Register a new user with a username and password."""
    if username in user_accounts:
        return f"User '{username}' already exists."
    user_accounts[username] = {"password": password, "projects": []}
    return f"User '{username}' successfully registered."

def submit_project(username: str, project_description: str, industry: str, project_source: str) -> str:
    """
    Submit a deployed AI project.
    Stores a description, industry, and project source.
    Immediately checks the project's history for misalignment incidents.
    """
    if username not in user_accounts:
        return f"User '{username}' is not registered."
    
    project = {
        "description": project_description,
        "industry": industry.lower(),
        "source": project_source
    }
    user_accounts[username]["projects"].append(project)
    
    # Immediately check for historical misalignment incidents.
    history_report = check_project_history(project)
    
    return f"Project submitted successfully for user '{username}'.\n{history_report}"

print(register_user("alice", "password123"))
print(submit_project(
    "alice", 
    "An AI diagnostic system for early disease detection.", 
    "Healthcare", 
    "https://github.com/alice/ai-diagnostic"
))

# %%
def list_projects(username: str) -> str:
    """Return a summary of all projects submitted by the user."""
    if username not in user_accounts:
        return f"User '{username}' is not registered."
    projects = user_accounts[username]["projects"]
    if not projects:
        return f"User '{username}' has not submitted any projects."
    
    output = f"Projects for {username}:\n"
    for idx, proj in enumerate(projects, start=1):
        output += (f"Project {idx}:\n"
                   f"  Industry: {proj['industry']}\n"
                   f"  Description: {proj['description']}\n"
                   f"  Source: {proj['source']}\n\n")
    return output

print(list_projects("alice"))

# %%
def query_graph(query: str) -> str:
    """
    Query our graph using the agent.
    Fallback gracefully if the agent fails.
    """
    llm = ChatGroq(
        model_name="llama3-70b-8192",
        api_key=os.environ.get("GROQ_API_KEY")
    )
    agent = initialize_agent(
        tools,
        llm,
        agent="zero-shot-react-description",
        verbose=True
    )
    try:
        return agent.run(query)
    except Exception as e:
        print("Graph query error:", e)
        return ""

def fetch_tweets(query: str) -> list:
    """
    Fetch tweets using Tweepy and the Twitter API v2 for the given query.
    Returns a list of dictionaries with keys: 'title', 'description', 'url'.
    """
    bearer_token = os.environ.get("TWITTER_BEARER_TOKEN")
    if not bearer_token:
        print("Error: TWITTER_BEARER_TOKEN is not set in the environment.")
        return []
    
    client = tweepy.Client(bearer_token=bearer_token)
    try:
        response = client.search_recent_tweets(query=query, tweet_fields=["id", "text"], max_results=10)
    except Exception as ex:
        print("Error fetching tweets:", ex)
        return []
    
    tweets = response.data
    if not tweets:
        return []
    
    results = []
    for tweet in tweets:
        tweet_id = tweet.id
        tweet_text = tweet.text
        tweet_url = f"https://twitter.com/i/web/status/{tweet_id}"
        results.append({
            "title": tweet_text[:50] + "..." if len(tweet_text) > 50 else tweet_text,
            "description": tweet_text,
            "url": tweet_url
        })
    return results

def compare_tweets_with_graph(query: str) -> str:
    """
    For a given query, fetch tweets via Tweepy and query our graph.
    Combine and return the outputs.
    """
    tweets = fetch_tweets(query)
    graph_result = query_graph(query)
    
    def is_similar(text1: str, text2: str) -> bool:
        return text1.lower() in text2.lower() or text2.lower() in text1.lower()
    
    combined_output = ""
    if graph_result and tweets:
        similar_found = any(is_similar(tweet["title"], graph_result) for tweet in tweets)
        if similar_found:
            combined_output += "Combined Result:\nGraph Data:\n" + graph_result + "\n\nTweets:\n"
            for tweet in tweets:
                combined_output += f"Title: {tweet['title']}\nURL: {tweet['url']}\n\n"
        else:
            combined_output += "Both sources have data but no strong similarity was found.\n\nGraph Data:\n" + graph_result + "\n\nTweets:\n"
            for tweet in tweets:
                combined_output += f"Title: {tweet['title']}\nURL: {tweet['url']}\n\n"
    elif graph_result:
        combined_output = "Graph Data:\n" + graph_result
    elif tweets:
        combined_output = "Tweets:\n" + "\n".join([f"Title: {tweet['title']} - URL: {tweet['url']}" for tweet in tweets])
    else:
        combined_output = "Fallback: No relevant information found in either source."
    
    return combined_output

# %%
def check_project_history(project: dict) -> str:
    """
    Check if the project's AI has previously misaligned by searching our graph and tweets.
    Returns a historical report if incidents are found; otherwise, a fallback message.
    """
    search_query = f"misalign {project['industry']} {project['description']}"
    graph_history = query_graph(search_query)
    tweets_history = fetch_tweets(search_query)
    
    if not graph_history and not tweets_history:
        return "No historical misalignment incidents found for this project."
    
    report = "Historical Misalignment Report:\n"
    if graph_history:
        report += f"Graph History:\n{graph_history}\n\n"
    if tweets_history:
        report += "Tweets History:\n"
        for tweet in tweets_history:
            report += f"Title: {tweet['title']}\nURL: {tweet['url']}\n\n"
    return report

def monitor_project(project: dict) -> str:
    """
    Monitor a specific project by running a history check.
    Returns a monitoring report.
    """
    try:
        return f"Monitoring Report for project in '{project['industry']}':\n" + check_project_history(project)
    except Exception as e:
        print(f"Error monitoring project in industry '{project['industry']}':", e)
        return "Fallback: Monitoring error for this project."

# %%
def send_email_alert(to_email: str, subject: str, message: str):
    """
    Sends an email alert using SMTP.
    Ensure that SMTP_SERVER, SMTP_PORT, EMAIL_USER, and EMAIL_PASS are set in your environment.
    """
    smtp_server = os.environ.get("SMTP_SERVER")
    smtp_port = os.environ.get("SMTP_PORT")
    email_user = os.environ.get("EMAIL_USER")
    email_pass = os.environ.get("EMAIL_PASS")
    
    if not (smtp_server and smtp_port and email_user and email_pass):
        print("SMTP credentials are not fully set.")
        return

    msg = MIMEText(message)
    msg["Subject"] = subject
    msg["From"] = email_user
    msg["To"] = to_email

    try:
        with smtplib.SMTP(smtp_server, int(smtp_port)) as server:
            server.starttls()
            server.login(email_user, email_pass)
            server.sendmail(email_user, to_email, msg.as_string())
        print(f"Email sent to {to_email}")
    except Exception as e:
        print("Error sending email:", e)

def play_alarm_sound():
    """
    Play an alarm sound using the playsound library.
    Ensure that an alarm sound file (e.g., 'alarm.mp3') is available in the specified path.
    """
    try:
        alarm_file = "/Users/sylvesterduah/Documents/Code/De_Alignment/data/alarm.mp3"
        if os.path.exists(alarm_file):
            playsound(alarm_file)
        else:
            print("Alarm sound file not found.")
    except Exception as e:
        print("Error playing alarm sound:", e)

def send_alert(username: str, project: dict, score: int, report: str):
    """
    Sends an alert by printing, sending email notifications, and playing an alarm sound.
    Assumes the username is the user's email and ADMIN_EMAIL is set for the overseer.
    """
    alert_message = (f"ALERT for user '{username}':\n"
                     f"Project in '{project['industry']}' has a risk score of {score}.\n"
                     f"Report:\n{report}\n")
    print(alert_message)
    
    # Send email to the user.
    user_email = username
    send_email_alert(user_email, "AI Misalignment Alert", alert_message)
    
    # Send email to the admin.
    admin_email = os.environ.get("ADMIN_EMAIL")
    if admin_email:
        send_email_alert(admin_email, "AI Misalignment Alert - Project Oversight", alert_message)
    else:
        print("ADMIN_EMAIL not set; skipping admin alert.")
    
    # Play an alarm sound.
    play_alarm_sound()


# %%
def calculate_risk_score(report: str) -> int:
    """
    Calculate a risk score based on keywords in the misalignment report.
    This is a simplified risk scoring mechanism.
    """
    keywords = {
        "critical": 5,
        "severe": 4,
        "misalign": 3,
        "error": 2,
        "unsafe": 3,
        "inconsistent": 2,
        "fault": 2,
        "issue": 1
    }
    score = 0
    report_lower = report.lower()
    for word, weight in keywords.items():
        score += report_lower.count(word) * weight
    return score

def vulnerability_scan(project: dict) -> str:
    """
    Simulate a vulnerability scan on the project source.
    For GitHub sources, assume a basic scan.
    """
    source = project.get("source", "").lower()
    if "github.com" in source:
        return "No known vulnerabilities detected on GitHub."
    else:
        return "Vulnerability scan not available for the provided source."

def send_alert(username: str, project: dict, score: int, report: str):
    """
    Simulate sending an alert if the risk score exceeds a threshold.
    In a production system, this could trigger an email or push notification.
    """
    print(f"ALERT for user '{username}': Project in '{project['industry']}' has a risk score of {score}.")
    print("Report:")
    print(report)
    print("-" * 50)

def enhanced_monitoring():
    """
    Extended monitoring that includes risk scoring, vulnerability scanning, 
    and alert notifications for each project.
    """
    ALERT_THRESHOLD = 10 
    alerts = []
    
    for username, info in user_accounts.items():
        for project in info["projects"]:
            print(f"Monitoring project for user '{username}' in industry '{project['industry']}'...")
            try:
                report = check_project_history(project)
            except Exception as e:
                print(f"Error checking history for user '{username}':", e)
                report = "Fallback: Monitoring error."
            risk = calculate_risk_score(report)
            vuln_result = vulnerability_scan(project)
            print(f"Vulnerability scan for project in '{project['industry']}': {vuln_result}")
            if risk >= ALERT_THRESHOLD:
                send_alert(username, project, risk, report)
                alerts.append((username, project, risk, report))
    
    if alerts:
        return f"Enhanced Monitoring Report: {len(alerts)} alerts triggered."
    else:
        return "Enhanced Monitoring Report: No high-risk misalignment issues detected."

def plot_alerts():
    """
    Plot a bar chart of misalignment alerts per industry.
    """
    alert_industries = []
    for username, info in user_accounts.items():
        for project in info["projects"]:
            report = check_project_history(project)
            keywords = ["misalign", "error", "fault", "unsafe", "inconsistent", "issue"]
            if any(keyword in report.lower() for keyword in keywords):
                alert_industries.append(project["industry"])
    if not alert_industries:
        print("No alerts to display.")
        return
    
    counts = Counter(alert_industries)
    industries = list(counts.keys())
    values = list(counts.values())
    
    plt.figure(figsize=(8, 5))
    plt.bar(industries, values, color='salmon')
    plt.xlabel("Industry")
    plt.ylabel("Number of Alerts")
    plt.title("Misalignment Alerts per Industry")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Run enhanced monitoring and display the dashboard.
print(enhanced_monitoring())
plot_alerts()

# %%
def global_monitoring_system():
    """
    Iterate over all registered projects and compile a global radar report,
    combining history check and cybersecurity enhancements.
    """
    alerts = []
    
    for username, info in user_accounts.items():
        for project in info["projects"]:
            print(f"Monitoring project for user '{username}' in industry '{project['industry']}'...")
            try:
                report = check_project_history(project)
            except Exception as e:
                print(f"Error monitoring project for '{username}':", e)
                report = "Fallback: Monitoring error."
            risk = calculate_risk_score(report)
            if risk >= 10:
                alerts.append((username, project, risk, report))
    
    if alerts:
        radar_report = "Global Radar Report: Potential AI misalignments detected in the following projects:\n"
        for username, project, risk, report in alerts:
            radar_report += (f"\nUser: {username}\n"
                             f"Industry: {project['industry']}\n"
                             f"Project Source: {project['source']}\n"
                             f"Risk Score: {risk}\n"
                             f"Report:\n{report}\n")
        return radar_report
    else:
        return "Global Radar Report: No significant misalignment issues detected across all projects."

print(global_monitoring_system())

# %%
refined_query = (
    "Find the ai_node from the 'ai_nodes' collection representing a person, AI, institution, "
    "or company with the highest connectivity (e.g., based on degree or pagerank). Explain why "
    "this entity is considered the most popular."
)
final_result = query_graph(refined_query)
print("Final Output (Popularity Analysis):")
print(final_result)