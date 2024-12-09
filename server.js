const express = require('express');
const cors = require('cors');
const fs = require('fs').promises;
const path = require('path');

const app = express();
const PORT = 3000;

// Enable CORS for all routes
app.use(cors({
  origin: 'http://localhost:5173', // Vite's default dev server port
  methods: ['GET', 'POST'],
  credentials: true
}));

// Add logging middleware
app.use((req, res, next) => {
  console.log(`${new Date().toISOString()} - ${req.method} ${req.url}`);
  next();
});

// Serve static files from outputs directory
app.use('/outputs', express.static(path.join(__dirname, 'outputs')));

// List directory contents
app.get('/outputs/:directory', async (req, res) => {
  try {
    const dirPath = path.join(__dirname, 'outputs', req.params.directory);
    const files = await fs.readdir(dirPath);
    res.json(files);
  } catch (error) {
    console.error(`Error reading directory ${req.params.directory}:`, error);
    res.status(500).send(`Error reading directory ${req.params.directory}`);
  }
});

app.get('/year-in-review', async (req, res) => {
  try {
    const filePath = path.join(__dirname, 'outputs', 'year_in_review_narrative.txt');
    console.log('Reading file:', filePath);
    const content = await fs.readFile(filePath, 'utf-8');
    res.type('text/plain').send(content);
  } catch (error) {
    console.error('Error reading year in review file:', error);
    res.status(500).send('Error reading year in review content');
  }
});

app.get('/research-summary', async (req, res) => {
  try {
    const filePath = path.join(__dirname, 'outputs', 'research_summary.csv');
    console.log('Reading file:', filePath);
    const content = await fs.readFile(filePath, 'utf-8');
    res.type('text/csv').send(content);
  } catch (error) {
    console.error('Error reading research summary file:', error);
    res.status(500).send('Error reading research summary');
  }
});

app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
  // Log the current directory and check if files exist
  console.log('Current directory:', __dirname);
  console.log('Outputs directory:', path.join(__dirname, 'outputs'));
}); 