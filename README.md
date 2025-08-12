# Image Editor UI

An interactive web interface for the image editing system with scribble and add modes.

## Architecture

This project consists of two parts:
- **Frontend**: React TypeScript application for the user interface
- **Backend**: Flask API server that runs the GPU-intensive diffusion models

## Features

### UI Components
1. **Dual Image Display**: Side-by-side boxes showing current image and edit preview
2. **Interactive Drawing**: Overlay canvas for scribble mode and bounding box selection
3. **Mode Switching**: Toggle between Add Mode and Scribble Mode
4. **Accept/Reject Workflow**: Preview edits before applying them
5. **Settings Panel**: Adjust generation parameters like seed, steps, and brush size

### Editing Modes
- **Add Mode**: Draw bounding box and specify what to add to that region
- **Scribble Mode**: Draw rough sketches that get transformed into detailed additions (like hair ribbons)

## Setup Instructions

### 1. Backend Setup (Server with GPU)

Navigate to the main project directory:
```bash
cd /path/to/box_edit
```

Install Python dependencies:
```bash
pip install -r api_requirements.txt
```

Start the Flask server:
```bash
python app.py
```

The server will start on `http://localhost:5000` and automatically load the diffusion models.

### 2. Frontend Setup (Local Computer)

Navigate to the React app directory:
```bash
cd image-editor-ui
```

Install Node.js dependencies:
```bash
npm install
```

**Important**: Update the API URL in `src/App.tsx` if your server is not on localhost:
```typescript
// Change this line to match your server address
const API_BASE = 'http://YOUR_SERVER_IP:5000/api';
```

Start the React development server:
```bash
npm start
```

The UI will open at `http://localhost:3000`

## Usage Guide

### Getting Started
1. Enter a prompt (e.g., "anime girl with long hair") and click **Generate**
2. Wait for the base image to appear in the left box
3. Choose your editing mode and start editing!

### Add Mode
1. Click **Add Mode** to enable bounding box selection
2. Draw a rectangle on the image where you want to add something
3. Enter what you want to add (e.g., "butterfly", "flower")
4. Click **Add** and wait for the preview
5. **Accept** or **Reject** the edit

### Scribble Mode  
1. Click **Scribble Mode** to enable drawing
2. Draw a rough sketch on the image
3. Enter what the scribble should become (e.g., "pink hair ribbon")
4. Adjust brush size if needed using the slider in the top bar
5. Click **Apply** and wait for the preview
6. **Accept** or **Reject** the edit

### Settings
- **Seed**: Controls randomness (same seed = same result)
- **Steps**: More steps = higher quality but slower generation
- **Brush Size**: Only appears in Scribble Mode

## API Endpoints

The Flask backend provides these endpoints:

### `POST /api/generate`
Generate a new base image
```json
{
  "prompt": "anime girl with long hair",
  "seed": 42,
  "num_inference_steps": 50
}
```

### `POST /api/edit/add`
Add elements using bounding box
```json
{
  "session_id": "uuid",
  "box": [x0, y0, x1, y1],
  "add_prompt": "butterfly",
  "num_inference_steps": 50
}
```

### `POST /api/edit/scribble`
Edit using scribble drawing
```json
{
  "session_id": "uuid",
  "scribble_image": "base64_image_data",
  "scribble_prompt": "pink hair ribbon",
  "num_inference_steps": 50
}
```

### `POST /api/accept_edit`
Accept the current edit preview
```json
{
  "session_id": "uuid"
}
```

### `POST /api/reject_edit`
Reject the current edit preview
```json
{
  "session_id": "uuid"
}
```

## Technical Details

### Frontend Stack
- **React 19** with TypeScript
- **Fabric.js** for canvas drawing and interaction
- **Axios** for API communication
- **CSS3** with responsive design

### Backend Stack
- **Flask** with CORS support
- **PyTorch** and **Diffusers** for AI models
- **ControlNet** for scribble-based editing
- **InstDiffEdit** for automatic attention masking

### Key Features
- **Session Management**: Each browser session maintains its own image state
- **Latent Caching**: Base images are stored as latents for faster editing
- **Hybrid Masking**: Combines ControlNet constraints with attention-based blending
- **Real-time Preview**: Edit results are shown before applying

## Troubleshooting

### Common Issues

**"Failed to generate image. Is the server running?"**
- Check if the Flask server is running on the correct port
- Verify the API_BASE URL in App.tsx matches your server address
- Check server logs for CUDA/GPU issues

**Canvas not responding in drawing modes**
- Ensure you have an image loaded first
- Try refreshing the page and generating a new image
- Check browser console for JavaScript errors

**Out of memory errors**
- Reduce batch size or image resolution in the server code
- Monitor GPU memory usage on the server
- Consider using CPU fallback for testing

**Slow generation times**
- Reduce num_inference_steps (try 20-30 for faster results)
- Ensure CUDA is properly configured on the server
- Consider using xformers for memory optimization

### Performance Tips
- Use lower step counts (20-30) for faster previews
- Generate base images once and reuse them for multiple edits
- Keep scribbles simple and clear for better results
- Use consistent seeds for reproducible results

## Development

To modify the UI:
1. Edit React components in `src/`
2. Styles are in `src/App.css`
3. API integration is in `src/App.tsx`

To modify the backend:
1. Edit `app.py` for API endpoints
2. Model logic is imported from existing project files
3. Add new endpoints following the existing pattern

## Credits

Built on top of the InstDiffEdit and ControlNet image editing system with hybrid attention masking for precise, localized edits.
