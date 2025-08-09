"""
Example 3: Memory Patterns with Agents
=======================================

This example demonstrates how to work with memory in agents, specifically
showcasing the WorkingMemoryAgent and memory management patterns.

Prerequisites:
- Set OPENROUTER_API_KEY environment variable
- Install arshai package
"""

import os
import asyncio
from typing import Dict, Any
from arshai.agents.working_memory import WorkingMemoryAgent
from arshai.core.interfaces.iagent import IAgentInput
from arshai.core.interfaces.illm import ILLMConfig
from arshai.llms.openrouter import OpenRouterClient


class InMemoryManager:
    """
    Simple in-memory storage for demonstration.
    
    In production, use Redis or a proper database.
    """
    
    def __init__(self):
        self.memories = {}
        self.access_count = {}
    
    async def store(self, data: Dict[str, Any]):
        """Store memory for a conversation."""
        conv_id = data.get("conversation_id")
        if conv_id:
            self.memories[conv_id] = data.get("working_memory", "")
            self.access_count[conv_id] = self.access_count.get(conv_id, 0) + 1
            print(f"  💾 [STORED] Memory for {conv_id} (Access count: {self.access_count[conv_id]})")
    
    async def retrieve(self, query: Dict[str, Any]):
        """Retrieve memory for a conversation."""
        conv_id = query.get("conversation_id")
        if conv_id and conv_id in self.memories:
            print(f"  📚 [RETRIEVED] Memory for {conv_id}")
            return [type('obj', (), {'working_memory': self.memories[conv_id]})()]
        print(f"  ❌ [NOT FOUND] No memory for {conv_id}")
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory storage statistics."""
        return {
            "total_conversations": len(self.memories),
            "conversations": list(self.memories.keys()),
            "access_counts": self.access_count
        }
    
    def show_memory(self, conv_id: str):
        """Display memory content for debugging."""
        if conv_id in self.memories:
            print(f"  📋 Memory for {conv_id}:")
            print(f"     {self.memories[conv_id]}")
        else:
            print(f"  ❌ No memory found for {conv_id}")


class ConversationSimulator:
    """Simulates a conversation to demonstrate memory patterns."""
    
    def __init__(self, memory_manager, memory_agent):
        self.memory_manager = memory_manager
        self.memory_agent = memory_agent
        self.conversation_turns = []
    
    async def add_interaction(self, conversation_id: str, interaction: str) -> str:
        """Add an interaction and update memory."""
        print(f"\n👤 User interaction: {interaction}")
        
        # Store this interaction
        self.conversation_turns.append(interaction)
        
        # Update memory with this interaction
        input_data = IAgentInput(
            message=interaction,
            metadata={"conversation_id": conversation_id}
        )
        
        # Process memory update
        result = await self.memory_agent.process(input_data)
        print(f"  🧠 Memory update status: {result}")
        
        return result
    
    def get_conversation_history(self) -> list:
        """Get the full conversation history."""
        return self.conversation_turns.copy()


async def main():
    """Demonstrate memory patterns with agents."""
    
    # Check for API key
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("⚠️  Please set OPENROUTER_API_KEY environment variable")
        return
    
    print("=" * 60)
    print("WORKING MEMORY AGENT DEMONSTRATION")
    print("=" * 60)
    
    # Initialize LLM client
    config = ILLMConfig(model="openai/gpt-4o-mini", temperature=0.5)
    llm_client = OpenRouterClient(config)
    
    # Create memory manager
    print("\n🔧 Setting up memory manager...")
    memory_manager = InMemoryManager()
    
    # Create memory agent
    print("🤖 Creating WorkingMemoryAgent...")
    memory_agent = WorkingMemoryAgent(
        llm_client=llm_client,
        memory_manager=memory_manager
    )
    
    # Test 1: Basic Memory Operations
    print("\n" + "=" * 40)
    print("TEST 1: Basic Memory Operations")
    print("=" * 40)
    
    conversation_id = "user_alice_session_001"
    
    interactions = [
        "My name is Alice and I work as a software engineer at TechCorp",
        "I'm interested in learning about machine learning for my current project",
        "Specifically, I need help with natural language processing techniques",
        "I prefer practical examples over theoretical explanations"
    ]
    
    simulator = ConversationSimulator(memory_manager, memory_agent)
    
    for i, interaction in enumerate(interactions, 1):
        print(f"\n--- Turn {i} ---")
        await simulator.add_interaction(conversation_id, interaction)
        
        # Show current memory state
        memory_manager.show_memory(conversation_id)
    
    # Test 2: Multiple Conversations
    print("\n" + "=" * 40)
    print("TEST 2: Multiple Conversations")
    print("=" * 40)
    
    # Second conversation
    conversation_2 = "user_bob_session_002"
    bob_interactions = [
        "I'm Bob, a product manager looking to understand AI capabilities",
        "My team is considering implementing chatbots for customer support"
    ]
    
    print("\n🔄 Starting second conversation...")
    for interaction in bob_interactions:
        await simulator.add_interaction(conversation_2, interaction)
    
    # Show memory for both conversations
    print("\n📊 Memory Status for All Conversations:")
    stats = memory_manager.get_stats()
    print(f"   Total conversations: {stats['total_conversations']}")
    print(f"   Conversation IDs: {stats['conversations']}")
    
    for conv_id in stats['conversations']:
        memory_manager.show_memory(conv_id)
    
    # Test 3: Error Handling
    print("\n" + "=" * 40)
    print("TEST 3: Error Handling")
    print("=" * 40)
    
    # Test without conversation_id
    print("\n🧪 Testing without conversation_id...")
    result = await memory_agent.process(IAgentInput(
        message="This message has no conversation ID",
        metadata={}
    ))
    print(f"   Result: {result}")
    
    # Test with empty metadata
    print("\n🧪 Testing with None metadata...")
    result = await memory_agent.process(IAgentInput(
        message="This message has None metadata",
        metadata=None
    ))
    print(f"   Result: {result}")
    
    # Test 4: Memory Agent without Memory Manager
    print("\n" + "=" * 40)
    print("TEST 4: Agent without Memory Manager")
    print("=" * 40)
    
    # Create agent without memory manager
    standalone_agent = WorkingMemoryAgent(
        llm_client=llm_client,
        memory_manager=None  # No storage
    )
    
    print("\n🧪 Testing agent without memory manager...")
    result = await standalone_agent.process(IAgentInput(
        message="User wants to learn Python programming",
        metadata={"conversation_id": "test_no_storage"}
    ))
    print(f"   Result: {result}")
    print("   Note: Memory was generated but not stored (no manager)")
    
    # Test 5: Advanced Memory Patterns
    print("\n" + "=" * 40)
    print("TEST 5: Advanced Memory Patterns")
    print("=" * 40)
    
    # Simulate a customer support conversation
    print("\n📞 Simulating customer support conversation...")
    support_conversation = "support_ticket_12345"
    
    support_interactions = [
        "Customer Jane Smith called about billing issue with invoice #4567",
        "Issue: Charged twice for the same service in March 2024",
        "Customer provided transaction IDs: TXN001, TXN002",
        "Resolution: Refunded duplicate charge of $99.99",
        "Customer satisfied, case closed"
    ]
    
    for i, interaction in enumerate(support_interactions, 1):
        print(f"\n📝 Support Log {i}: {interaction}")
        result = await memory_agent.process(IAgentInput(
            message=interaction,
            metadata={"conversation_id": support_conversation}
        ))
        print(f"   Memory Status: {result}")
    
    # Show final memory state
    print("\n📋 Final Support Case Memory:")
    memory_manager.show_memory(support_conversation)
    
    # Test 6: Memory Retrieval Simulation
    print("\n" + "=" * 40)
    print("TEST 6: Memory Retrieval Simulation")
    print("=" * 40)
    
    # Simulate retrieving memory for a returning customer
    print("\n🔍 Simulating memory retrieval for returning customer...")
    
    # Add new interaction that should reference previous memory
    result = await memory_agent.process(IAgentInput(
        message="Alice called back asking about implementation timeline for the ML project we discussed",
        metadata={"conversation_id": conversation_id}  # Same ID as Alice's original conversation
    ))
    
    print(f"   Update Status: {result}")
    print("\n📋 Updated Memory (should include new context):")
    memory_manager.show_memory(conversation_id)
    
    # Final Statistics
    print("\n" + "=" * 60)
    print("FINAL MEMORY STATISTICS")
    print("=" * 60)
    
    final_stats = memory_manager.get_stats()
    print(f"📊 Total Conversations Tracked: {final_stats['total_conversations']}")
    print(f"🔄 Access Patterns:")
    for conv_id, count in final_stats['access_counts'].items():
        print(f"   {conv_id}: {count} updates")
    
    print("\n✅ Memory patterns demonstration completed!")
    print("\nKey Takeaways:")
    print("• WorkingMemoryAgent manages conversation context automatically")
    print("• Returns 'success' or 'error: description' for status tracking")
    print("• Requires conversation_id in metadata")
    print("• Works with any memory manager (Redis, DB, etc.)")
    print("• Gracefully handles missing memory manager")
    print("• Builds contextual memory over multiple interactions")


if __name__ == "__main__":
    asyncio.run(main())