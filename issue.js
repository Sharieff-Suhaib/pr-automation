import dotenv from "dotenv";
dotenv.config();

const repoUrl = "https://github.com/Small-Fish-Dev/shrimple_character_controller/issues?utm_source=chatgpt.com";
// https://github.com/public-apis/public-apis
const url = new URL(repoUrl);

const owner = url.pathname.split("/")[1];
const repo = url.pathname.split("/")[2];

const response = await fetch(
    `https://api.github.com/repos/${owner}/${repo}/issues`,
    {
        headers: {
            "Accept": "application/vnd.github+json",
            "Authorization": `Bearer ${process.env.GITHUB_TOKEN}`,
            "X-GitHub-Api-Version": "2026-03-10"
        }
    }
);
const issues = await response.json();
// console.log(issues);

for (const issue of issues) {
    console.log(`Issue #${issue.number}: ${issue.title}`);
    console.log(`Body: ${issue.body}`);
    console.log(`State: ${issue.state}`);
}