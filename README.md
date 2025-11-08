# Talent Match Dashboard  
*Built during airport transits, caffeine highs, and a lot of trial and error.*

## Introduction  
This project started under slightly chaotic circumstances.  
I was traveling overseas when I received this case study, with limited internet access and only a few short windows to work on my laptop. The HR team was kind enough to give me an extended deadline, and I’m genuinely grateful for that.  

So, I built this entire dashboard from scratch (during **a 16-hour layover at Changi Airport** and the **three days after I landed**).  
It’s also my **first time using Streamlit, Supabase and OpenRouter**, so everything you see here was built while learning both tools in real time, sorry not sorry if you see some odd syntax in the project.

## The Objective  
The main goal of this exploration is simple:  
To understand what makes top employees *top performers.*  

I wanted to analyze different aspects from psychometric profiles, IQ/GTQ scores, and competencies to overall performance ratings and identify which patterns separate average performers from the best ones.  
The challenge was to turn complex data into something **human-friendly.**  
This dashboard was designed specifically so that **non-technical people (like HR or management)** can understand insights without reading SQL queries or sifting through raw spreadsheets. (Even tho

## Live Dashboard  
You can explore the deployed app here:  
👉 [**Talent Match Dashboard (Streamlit)**](https://talent-match-starter-alvnmlndr.streamlit.app/)

## What the Dashboard Does  
- Displays **top 5 divisions, departments, and education levels** based on performance and match scores  
- Lets users **filter** by company, division, education, and more to explore specific groups  
- Provides a **benchmarking tool** — select high performers and compare others against their profile  
- Generates an **AI-based job profile** using OpenRouter, summarizing what an “ideal candidate” might look like based on real data  
- All built in a **Streamlit web app**, kept intentionally clean and simple for non-technical readers  

## Tools and Tech Stack  
Built entirely from scratch with:  
- **Python** – for analysis and data processing  
- **Pandas + SQLAlchemy** – for data manipulation and database connectivity  
- **Supabase (PostgreSQL)** – for database hosting and SQL logic  
- **Streamlit** – for interactive dashboards  
- **OpenRouter API** – for AI-generated role profiles  
- **Excel** – because every great data project starts with a spreadsheet  

## Time Spent  
Approximate working time: **35 hours**  
This includes data preparation, setting up Supabase, writing SQL, building the dashboard, testing it, and learning two new frameworks on the fly.


## Key Takeaways  
- You can learn new frameworks much faster when there’s a deadline.  
- Simplicity always wins — data means little if people can’t interpret it.  
- And sometimes, the best projects happen in airports with bad Wi-Fi and a stubborn laptop battery.
