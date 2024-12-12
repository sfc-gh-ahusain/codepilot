import asyncio
import time

import cost_estimator
import faiss_utils
import llm_utils
import code_pilot_utils

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text
from rich.layout import Layout
from rich.live import Live


class CodePilotCLI:

    def __init__(self, config, codepilot):
        self.config = config
        self.faiss_index = codepilot.faiss_index
        self.index_file_mapping = codepilot.index_file_mapping
        self.llm_model = codepilot.llm_model
        self.cost_estimates = codepilot.cost_estimates
        self.console = Console()
        self.devtools = None

        # Set up the layout
        self.layout = Layout(name="root")
        self.layout.split(
            Layout(name="logo", size=6),  # Fixed size for the logo
            Layout(name="content", ratio=1)  # The content area is flexible and scrollable
        )

        # Initialize the response text as a rich Text object
        self.response_text = Text("", style="green")


    def display_logo(self):
        code_pilot_logo =     logo = """
    CCCCC   OOO   DDDD    EEEEE       PPPP    III   L       OOO   TTTTT
    C      O   O  D   D   E           P   P    I    L      O   O    T
    C      O   O  D   D   EEEE        PPPP     I    L      O   O    T
    C      O   O  D   D   E           P        I    L      O   O    T
    CCCCC   OOO   DDDD    EEEEE       P       III   LLLLL   OOO     T
    """
        self.layout["logo"].update(Text.from_markup(f"[bold cyan]{code_pilot_logo}[/bold cyan]"))


    def display_welcome_message(self):
        # Display welcome message in the content area
        self.layout["content"].update(Text("\nWelcome to CodePilot!\n", style="bold yellow"))


    def render_layout(self):
        self.console.print(self.layout)


    def get_user_query(self):
        while True:
            query = Prompt.ask("Enter your query ('quit' or 'stop' to exit)", default="")
            if query.strip().lower() not in [""]:
                return query
            else:
                self.console.print("[bold red]Please enter a valid query.[/bold red]")


    # This method will print each word with a delay inside the response_text to simulate a typing effect
    async def display_response_with_delay(self, user_query, response, actual_cost):
        # Create response text object that will be updated progressively
        self.response_text = Text("", style="green")

        # Create the panel only once
        panel = Panel(self.response_text, title="Query/Response", subtitle="Live Interaction", expand=True)

        # During Live, avoid rendering the panel through layout
        with Live(panel, console=self.console, refresh_per_second=10) as live:
            self.response_text.append(f"Query: ", style="bold yellow")
            self.response_text.append(f"{user_query}\n", style="white")
            self.response_text.append("Response: ", style="bold yellow")

            # Simulate typing effect by appending words one by one
            for word in response.split():
                self.response_text.append(word + " ")
                live.update(panel)
                # Delay to simulate typing effect
                await asyncio.sleep(0.2)

            # Once response is done, append the "Actual Cost" to the same response panel
            cost_text = Text("\n")
            cost_text.append("Actual Cost: ", style="bold yellow")  # Apply bold yellow to the label
            cost_text.append(f"${actual_cost:.6f}", style="bold red")  # Apply green to the cost            
            self.response_text.append(cost_text)  # Append the cost to the response text
            live.update(panel)  # Update the panel again with the cost included

        # After the live session ends, ensure the panel is updated in the layout
        self.layout["content"].update(panel)
        self.console.print(self.layout)  # Print the updated layout with cost included


    async def query_loop(self):
        while True:
            # Step 2: Prompt user to enter a query
            user_query = self.get_user_query()
            # Check if the user wants to exit
            if user_query.lower() in ["quit", "stop"]:
                self.console.print("[bold red]Exiting CodePilot CLI. Goodbye![/bold red]")
                break

            # Process the query to extract entities and keywords
            entities, keywords = faiss_utils.process_query(user_query)
            self.config.logger.debug(f"Extracted entities: {entities}, keywords: {keywords}")
            user_query_embedding = faiss_utils.get_query_embedding(self.config, user_query)

            distances, indices = faiss_utils.query_faiss_index(
                self.faiss_index, user_query_embedding, 5
                )

            # Fetch results and prepare the context
            insights, index_files = code_pilot_utils.fetch_insights(self.config, indices, self.index_file_mapping)
            if insights:
                # Estimate cost for running the query, let user choose to proceed
                estimated_cost = cost_estimator.estimate_query_cost(self.config, index_files, self.cost_estimates) 

                self.console.print(f"""
                                   \n[bold yellow]Estimated Cost:[/bold yellow] $[green]{estimated_cost:.6f}[/green]
                                   """)

                proceed = Prompt.ask("Do you want to proceed with this cost?", choices=["yes", "no"])
                if proceed.lower() == "yes":
                    # Process query using Codox
                    self.console.print("[bold cyan]Processing query...[/bold cyan]")
                
                    # Construct the prompt dynamically using the user's question and the insights
                    input_prompt = code_pilot_utils.construct_prompt(insights, user_query, entities, keywords)

                    # Consult with LLM and generate response
                    if input_prompt:
                        response = llm_utils.generate_response(self.config, input_prompt, self.llm_model)
                        actual_cost = cost_estimator.calculate_actual_cost(
                            self.config.embedding_model, estimated_cost, response
                            )

                        await self.display_response_with_delay(user_query, response, actual_cost)
                else:
                    self.console.print("[bold red]Query cancelled.[/bold red]")
            else:
                self.console.print(f"""
                                   \n[bold yellow]Something went wrong. Try again later.[/bold yellow]
                                   """)


    async def run(self):
        self.console.clear()
        self.display_logo()
        self.display_welcome_message()
        self.render_layout()
        await self.query_loop()
