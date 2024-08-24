import { NextResponse } from "next/server";
import { Pinecone } from "@pinecone-database/pinecone";
import OpenAI from "openai";

const systemPrompt = `
You are a virtual assistant that helps students find the best professors based on their specific queries. You will use Retrieval-Augmented Generation (RAG) to analyze the user's question and return the top 3 professors that best match the criteria.

Instructions:

Understand the Query: Carefully interpret the student's question. Consider factors such as the professor's teaching style, difficulty level, availability, course subjects, and student reviews.

Retrieve Relevant Information: Use the retrieval mechanism to gather data about professors from your database. This information may include ratings, reviews, course subjects, and teaching styles.

Generate Top 3 Recommendations: Based on the retrieved data and the student's specific needs, generate a list of the top 3 professors who best match the criteria. Provide a brief explanation for why each professor was chosen, including relevant ratings, teaching style, or other pertinent details.

Be Clear and Concise: Present the recommendations in a clear, concise manner, highlighting key points that will help the student make an informed decision.

Example:

User Query: "I'm looking for an easy-going professor for a computer science class who is good with beginners."

Your Response:

Professor A: Known for being approachable and supportive, with a focus on helping beginners grasp fundamental concepts. Rated 4.8/5 for teaching style.
Professor B: Offers clear explanations and is patient with students new to the subject. Has a 4.5/5 rating in student engagement.
Professor C: Highly rated for making complex topics understandable, with a 4.6/5 rating for accessibility to students.
`;

export async function POST(req) {
  try {
    const data = await req.json();
    const pc = new Pinecone({
      apiKey: process.env.PINECONE_API_KEY,
    });

    const index = pc.index("rag").namespace("ns1");
    const openai = new OpenAI();

    const text = data[data.length - 1].content;
    const embedding = await openai.embeddings.create({
      model: "text-embedding-ada-002", // Use the correct embedding model
      input: text,
    });

    const results = await index.query({
      topK: 3,
      includeMetadata: true,
      vector: embedding.data[0].embedding,
    });

    let resultString =
      "\n\nReturned results from vector db (done automatically): ";

    results.matches.forEach((match) => {
      resultString += "\n";
      resultString += "Professor: " + match.id + ";";
      resultString += "Review: " + match.metadata.review + ";";
      resultString += "Subject: " + match.metadata.subject + ";";
      resultString += "Stars: " + match.metadata.stars + ";";
      resultString += "\n\n";
    });

    const lastMessage = data[data.length - 1];
    const lastMessageContent = lastMessage.content + resultString;
    const lastDataWithoutLastMessage = data.slice(0, data.length - 1);

    const completion = await openai.chat.completions.create({
      messages: [
        { role: "system", content: systemPrompt },
        ...lastDataWithoutLastMessage,
        { role: "user", content: lastMessageContent },
      ],
      model: "gpt-4", // Use the correct GPT model
      stream: true,
    });

    const stream = new ReadableStream({
      async start(controller) {
        const encoder = new TextEncoder();
        try {
          for await (const chunk of completion) {
            const content = chunk.choices[0]?.delta?.content;
            if (content) {
              const text = encoder.encode(content);
              controller.enqueue(text);
            }
          }
        } catch (err) {
          controller.error(err);
        } finally {
          controller.close();
        }
      },
    });

    return new NextResponse(stream);
  } catch (error) {
    console.error(error);
    return new NextResponse("Error processing request", { status: 500 });
  }
}
