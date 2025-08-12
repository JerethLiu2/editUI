import React, { useState, useRef, useEffect } from 'react';
import axios, { AxiosResponse } from 'axios';
import { Canvas, Rect, PencilBrush, Circle, Triangle } from 'fabric';
import './App.css';

interface SessionData {
  sessionId: string;
  currentImage: string | null;
  editPreview: string | null;
}

type Mode = 'none' | 'add' | 'scribble';

interface BoundingBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

function App() {
  // State management
  const [session, setSession] = useState<SessionData>({
    sessionId: '',
    currentImage: null,
    editPreview: null
  });
  
  const [prompt, setPrompt] = useState('anime girl with long hair');
  const [mode, setMode] = useState<Mode>('none');
  const [isGenerating, setIsGenerating] = useState(false);
  const [isEditing, setIsEditing] = useState(false);
  
  // Settings
  const [seed, setSeed] = useState(-1);
  const [numSteps, setNumSteps] = useState(50);
  const [brushSize, setBrushSize] = useState(5);
  const [brushColor, setBrushColor] = useState('#FF0000');
  const [addDrawMode, setAddDrawMode] = useState<'pen' | 'rectangle' | 'circle' | 'triangle'>('pen');
  
  // Mode-specific inputs
  const [addPrompt, setAddPrompt] = useState('');
  const [scribblePrompt, setScribblePrompt] = useState('');
  const [boundingBox, setBoundingBox] = useState<BoundingBox | null>(null);
  
  // Canvas refs
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const fabricCanvasRef = useRef<Canvas | null>(null);
  
  // API base URL - change this to your server address
  const API_BASE = 'http://150.65.90.114:5000/api';

  useEffect(() => {
    // Initialize fabric canvas for drawing when in scribble mode
    if (canvasRef.current && mode === 'scribble') {
      // Always recreate canvas when entering scribble mode
      if (fabricCanvasRef.current) {
        fabricCanvasRef.current.dispose();
      }
      
      const canvas = new Canvas(canvasRef.current, {
        width: 512,
        height: 512,
        backgroundColor: 'transparent',
        renderOnAddRemove: true,
      });
      fabricCanvasRef.current = canvas;
      
      // Configure brush settings with better visibility
      const brush = new PencilBrush(canvas);
      brush.width = brushSize * 2;
      brush.color = brushColor;
      canvas.freeDrawingBrush = brush;
      
      // Enable drawing mode
      canvas.isDrawingMode = true;
      canvas.selection = false;
      canvas.renderAll(); // Force render
    }

    // Cleanup when leaving scribble mode
    return () => {
      if (fabricCanvasRef.current && mode !== 'scribble') {
        try {
          fabricCanvasRef.current.dispose();
        } catch (error) {
          console.warn('Error disposing scribble canvas:', error);
        } finally {
          fabricCanvasRef.current = null;
        }
      }
    };
  }, [mode]);

  // Update brush settings when they change
  useEffect(() => {
    if (fabricCanvasRef.current && mode === 'scribble') {
      const canvas = fabricCanvasRef.current;
      
      // Normal drawing mode
      const brush = new PencilBrush(canvas);
      brush.width = brushSize * 2;
      brush.color = brushColor;
      canvas.freeDrawingBrush = brush;
      canvas.isDrawingMode = true;
      
      canvas.renderAll();
    }
  }, [brushSize, brushColor, mode]);

  // Handle mode changes for add mode drawing
  useEffect(() => {
    if (mode === 'add' && canvasRef.current) {
      // Always recreate canvas when entering add mode
      if (fabricCanvasRef.current) {
        fabricCanvasRef.current.dispose();
      }
      
      const canvas = new Canvas(canvasRef.current, {
        width: 512,
        height: 512,
        backgroundColor: 'transparent',
        renderOnAddRemove: true,
      });
      fabricCanvasRef.current = canvas;
      
      setupAddModeDrawing(canvas);
    }

    // Cleanup when leaving add mode
    return () => {
      if (fabricCanvasRef.current && mode !== 'add' && mode !== 'scribble') {
        try {
          fabricCanvasRef.current.dispose();
        } catch (error) {
          console.warn('Error disposing canvas:', error);
        } finally {
          fabricCanvasRef.current = null;
        }
      }
    };
  }, [mode]);

  // Update add mode drawing when tool or settings change
  useEffect(() => {
    if (mode === 'add' && fabricCanvasRef.current) {
      setupAddModeDrawing(fabricCanvasRef.current);
    }
  }, [addDrawMode, brushColor, brushSize, mode]);

  const setupAddModeDrawing = (canvas: Canvas) => {
    // Clear any existing event listeners
    canvas.off('mouse:down');
    canvas.off('mouse:move');
    canvas.off('mouse:up');

    if (addDrawMode === 'pen') {
      // Free drawing mode
      const brush = new PencilBrush(canvas);
      brush.width = Math.min(brushSize, 10);
      brush.color = brushColor;
      canvas.freeDrawingBrush = brush;
      canvas.isDrawingMode = true;
      canvas.selection = false;
    } else {
      // Shape drawing mode
      canvas.isDrawingMode = false;
      canvas.selection = false;
      
      let isDown = false;
      let origX = 0;
      let origY = 0;
      let currentShape: any = null;

      canvas.on('mouse:down', (o: any) => {
        isDown = true;
        const pointer = canvas.getPointer(o.e);
        origX = pointer.x;
        origY = pointer.y;
        
        // Remove existing shape
        if (currentShape) {
          canvas.remove(currentShape);
        }
        
        // Create new shape based on selected tool
        if (addDrawMode === 'rectangle') {
          currentShape = new Rect({
            left: origX,
            top: origY,
            originX: 'left',
            originY: 'top',
            width: 1,
            height: 1,
            fill: 'transparent',
            stroke: brushColor,
            strokeWidth: Math.min(brushSize, 10),
            selectable: false
          });
        } else if (addDrawMode === 'circle') {
          currentShape = new Circle({
            left: origX,
            top: origY,
            originX: 'left',
            originY: 'top',
            radius: 1,
            fill: 'transparent',
            stroke: brushColor,
            strokeWidth: Math.min(brushSize, 10),
            selectable: false
          });
        } else if (addDrawMode === 'triangle') {
          currentShape = new Triangle({
            left: origX,
            top: origY,
            originX: 'left',
            originY: 'top',
            width: 1,
            height: 1,
            fill: 'transparent',
            stroke: brushColor,
            strokeWidth: Math.min(brushSize, 10),
            selectable: false
          });
        }
        
        if (currentShape) {
          canvas.add(currentShape);
        }
      });

      canvas.on('mouse:move', (o: any) => {
        if (!isDown || !currentShape) return;
        
        const pointer = canvas.getPointer(o.e);
        const width = Math.abs(pointer.x - origX);
        const height = Math.abs(pointer.y - origY);
        
        if (addDrawMode === 'rectangle' || addDrawMode === 'triangle') {
          currentShape.set({
            width: width,
            height: height,
            left: pointer.x > origX ? origX : pointer.x,
            top: pointer.y > origY ? origY : pointer.y
          });
        } else if (addDrawMode === 'circle') {
          const radius = Math.max(width, height) / 2;
          currentShape.set({
            radius: radius,
            left: origX - radius,
            top: origY - radius
          });
        }
        
        canvas.renderAll();
      });

      canvas.on('mouse:up', () => {
        isDown = false;
      });
    }
    
    canvas.renderAll();
  };

  const generateImage = async () => {
    setIsGenerating(true);
    try {
      const response = await axios.post(`${API_BASE}/generate`, {
        prompt,
        seed,
        num_inference_steps: numSteps,
        session_id: session.sessionId || undefined
      });
      
      if (response.data.success) {
        setSession({
          sessionId: response.data.session_id,
          currentImage: response.data.image,
          editPreview: null
        });
      } else {
        alert('Error generating image: ' + response.data.error);
      }
    } catch (error) {
      console.error('Generation error:', error);
      alert('Failed to generate image. Is the server running?');
    }
    setIsGenerating(false);
  };

  const executeEdit = async () => {
    if (!session.sessionId || !session.currentImage) {
      alert('Generate an image first!');
      return;
    }

    setIsEditing(true);
    try {
      let response: AxiosResponse<any>;
      
      if (mode === 'add' && addPrompt && fabricCanvasRef.current) {
        // Get add canvas data and convert transparent areas to white for backend
        const canvas = fabricCanvasRef.current;
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = 512;
        tempCanvas.height = 512;
        const tempCtx = tempCanvas.getContext('2d')!;
        
        // Fill with white background first
        tempCtx.fillStyle = '#FFFFFF';
        tempCtx.fillRect(0, 0, 512, 512);
        
        // Draw the fabric canvas on top (transparent areas will remain white)
        const fabricDataUrl = canvas.toDataURL({ format: 'png', multiplier: 1 });
        const img = new Image();
        
        await new Promise<void>((resolve) => {
          img.onload = () => {
            tempCtx.drawImage(img, 0, 0);
            resolve();
          };
          img.src = fabricDataUrl;
        });
        
        const addDataUrl = tempCanvas.toDataURL('image/png');
        const addBase64 = addDataUrl.split(',')[1];
        
        response = await axios.post(`${API_BASE}/edit/add`, {
          session_id: session.sessionId,
          add_image: addBase64,
          add_prompt: addPrompt,
          num_inference_steps: numSteps
        });
      } else if (mode === 'scribble' && scribblePrompt && fabricCanvasRef.current) {
        // Get scribble canvas data and convert transparent areas to white for backend
        const canvas = fabricCanvasRef.current;
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = 512;
        tempCanvas.height = 512;
        const tempCtx = tempCanvas.getContext('2d')!;
        
        // Fill with white background first
        tempCtx.fillStyle = '#FFFFFF';
        tempCtx.fillRect(0, 0, 512, 512);
        
        // Draw the fabric canvas on top (transparent areas will remain white)
        const fabricDataUrl = canvas.toDataURL({ format: 'png', multiplier: 1 });
        const img = new Image();
        
        await new Promise<void>((resolve) => {
          img.onload = () => {
            tempCtx.drawImage(img, 0, 0);
            resolve();
          };
          img.src = fabricDataUrl;
        });
        
        const scribbleDataUrl = tempCanvas.toDataURL('image/png');
        const scribbleBase64 = scribbleDataUrl.split(',')[1];
        
        response = await axios.post(`${API_BASE}/edit/scribble`, {
          session_id: session.sessionId,
          scribble_image: scribbleBase64,
          scribble_prompt: scribblePrompt,
          num_inference_steps: numSteps
        });
      } else {
        alert('Please set up the edit parameters first!');
        setIsEditing(false);
        return;
      }
      
      if (response.data.success) {
        setSession(prev => ({
          ...prev,
          editPreview: response.data.image
        }));
      } else {
        alert('Error during edit: ' + response.data.error);
      }
    } catch (error) {
      console.error('Edit error:', error);
      alert('Failed to execute edit. Check server logs.');
    }
    setIsEditing(false);
  };

  const acceptEdit = async () => {
    if (!session.editPreview) return;
    
    try {
      const response = await axios.post(`${API_BASE}/accept_edit`, {
        session_id: session.sessionId
      });
      
      if (response.data.success) {
        setSession(prev => ({
          ...prev,
          currentImage: prev.editPreview,
          editPreview: null
        }));
        
        // Safely dispose of canvas before mode change
        if (fabricCanvasRef.current) {
          fabricCanvasRef.current.dispose();
          fabricCanvasRef.current = null;
        }
        
        setMode('none');
      }
    } catch (error) {
      console.error('Accept error:', error);
      alert('Failed to accept edit');
    }
  };

  const rejectEdit = async () => {
    try {
      await axios.post(`${API_BASE}/reject_edit`, {
        session_id: session.sessionId
      });
      
      // Just clear the preview, keep canvas and mode for easy retry
      setSession(prev => ({
        ...prev,
        editPreview: null
      }));
      
      // Keep canvas and mode intact so user can try again immediately
    } catch (error) {
      console.error('Reject error:', error);
    }
  };

  const clearCanvas = () => {
    if (fabricCanvasRef.current) {
      try {
        fabricCanvasRef.current.clear();
      } catch (error) {
        console.warn('Error clearing canvas:', error);
      }
    }
    setBoundingBox(null);
  };

  const toggleMode = (newMode: Mode) => {
    if (mode === newMode) {
      // Deactivating current mode - safely dispose canvas first
      if (fabricCanvasRef.current) {
        try {
          fabricCanvasRef.current.dispose();
        } catch (error) {
          console.warn('Error disposing canvas during mode toggle:', error);
        } finally {
          fabricCanvasRef.current = null;
        }
      }
      setMode('none');
    } else {
      setMode(newMode);
      clearCanvas();
    }
  };

  return (
    <div className="App">
      {/* Options Bar */}
      <div className="options-bar">
        {mode === 'scribble' && (
          <>
            {/* Color Palette */}
            <div className="option-group">
              <label>Colors:</label>
              <div className="color-palette">
                {['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF', '#FFA500', '#000000'].map(color => (
                  <button
                    key={color}
                    className={`color-btn ${brushColor === color ? 'active' : ''}`}
                    style={{ backgroundColor: color }}
                    onClick={() => setBrushColor(color)}
                  />
                ))}
              </div>
            </div>
            
            {/* Clear Button */}
            <div className="option-group">
              <button
                className="clear-btn"
                onClick={clearCanvas}
              >
                Clear Canvas
              </button>
            </div>
            
            {/* Brush Size Slider */}
            <div className="option-group">
              <label>Brush Size:</label>
              <input
                type="range"
                min="1"
                max="50"
                value={brushSize}
                onChange={(e) => setBrushSize(parseInt(e.target.value))}
                className="brush-slider"
              />
              <span>{brushSize}px</span>
            </div>
          </>
        )}
        {mode === 'add' && (
          <>
            {/* Color Palette for Add Mode (no white) */}
            <div className="option-group">
              <label>Colors:</label>
              <div className="color-palette">
                {['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF', '#FFA500', '#000000'].map(color => (
                  <button
                    key={color}
                    className={`color-btn ${brushColor === color ? 'active' : ''}`}
                    style={{ backgroundColor: color }}
                    onClick={() => setBrushColor(color)}
                  />
                ))}
              </div>
            </div>
            
            {/* Drawing Tools */}
            <div className="option-group">
              <label>Tools:</label>
              <div className="tool-palette">
                <button
                  className={`tool-btn ${addDrawMode === 'pen' ? 'active' : ''}`}
                  onClick={() => setAddDrawMode('pen')}
                  title="Free drawing"
                >
                  ✏️
                </button>
                <button
                  className={`tool-btn ${addDrawMode === 'rectangle' ? 'active' : ''}`}
                  onClick={() => setAddDrawMode('rectangle')}
                  title="Rectangle"
                >
                  ⬜
                </button>
                <button
                  className={`tool-btn ${addDrawMode === 'circle' ? 'active' : ''}`}
                  onClick={() => setAddDrawMode('circle')}
                  title="Circle"
                >
                  ⭕
                </button>
                <button
                  className={`tool-btn ${addDrawMode === 'triangle' ? 'active' : ''}`}
                  onClick={() => setAddDrawMode('triangle')}
                  title="Triangle"
                >
                  🔺
                </button>
              </div>
            </div>
            
            {/* Clear Button */}
            <div className="option-group">
              <button
                className="clear-btn"
                onClick={clearCanvas}
              >
                Clear Canvas
              </button>
            </div>
            
            {/* Line Thickness */}
            <div className="option-group">
              <label>Line Thickness:</label>
              <input
                type="range"
                min="1"
                max="10"
                value={Math.min(brushSize, 10)}
                onChange={(e) => setBrushSize(parseInt(e.target.value))}
                className="brush-slider"
              />
              <span>{Math.min(brushSize, 10)}px</span>
            </div>
          </>
        )}
      </div>

      {/* Main Content */}
      <div className="main-content">
        {/* Toolbar */}
        <div className="toolbar">
          {/* Prompt Input */}
          <div className="prompt-section">
            <label>Prompt:</label>
            <textarea
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              className="prompt-input"
              placeholder="Enter your prompt..."
              rows={4}
            />
            <button 
              onClick={generateImage}
              disabled={isGenerating}
              className="generate-btn"
            >
              {isGenerating ? 'Generating...' : 'Generate'}
            </button>
          </div>

          {/* Generation Settings */}
          <div className="generation-settings">
            <div className="setting-group">
              <label>Seed:</label>
              <input
                type="number"
                value={seed}
                onChange={(e) => setSeed(parseInt(e.target.value))}
                className="setting-input"
              />
            </div>
            <div className="setting-group">
              <label>Steps:</label>
              <input
                type="number"
                value={numSteps}
                min="1"
                max="100"
                onChange={(e) => setNumSteps(parseInt(e.target.value))}
                className="setting-input"
              />
            </div>
          </div>

          {/* Mode Toggles */}
          <div className="mode-section">
            <button
              onClick={() => toggleMode('add')}
              className={`mode-btn ${mode === 'add' ? 'active' : ''}`}
              disabled={!session.currentImage}
            >
              Add Mode
            </button>
            <button
              onClick={() => toggleMode('scribble')}
              className={`mode-btn ${mode === 'scribble' ? 'active' : ''}`}
              disabled={!session.currentImage}
            >
              Scribble Mode
            </button>
          </div>

          {/* Mode-specific Controls */}
          {mode === 'add' && (
            <div className="add-controls">
              <input
                type="text"
                value={addPrompt}
                onChange={(e) => setAddPrompt(e.target.value)}
                placeholder="What to add..."
                className="add-prompt-input"
              />
              <button
                onClick={executeEdit}
                disabled={!addPrompt || isEditing}
                className="execute-btn"
              >
                {isEditing ? 'Adding...' : 'Add'}
              </button>
            </div>
          )}

          {mode === 'scribble' && (
            <div className="scribble-controls">
              <input
                type="text"
                value={scribblePrompt}
                onChange={(e) => setScribblePrompt(e.target.value)}
                placeholder="What should the scribble become..."
                className="scribble-prompt-input"
              />
              <button
                onClick={executeEdit}
                disabled={!scribblePrompt || isEditing}
                className="execute-btn"
              >
                {isEditing ? 'Processing...' : 'Apply'}
              </button>
            </div>
          )}
        </div>

        {/* Image Display Area */}
        <div className="image-container">
          {/* Left Box - Current Image */}
          <div className="image-box">
            <h3>Current Image</h3>
            <div className="image-wrapper">
              {session.currentImage ? (
                <img 
                  src={`data:image/png;base64,${session.currentImage}`}
                  alt="Current"
                  className="display-image"
                />
              ) : (
                <div className="placeholder">Generate an image to start</div>
              )}
              
              {/* Overlay Canvas for Drawing */}
              {(mode === 'scribble' || mode === 'add') && session.currentImage && (
                <canvas
                  ref={canvasRef}
                  className="drawing-canvas"
                />
              )}
            </div>
          </div>

          {/* Right Box - Edit Preview */}
          <div className="image-box">
            <h3>Edit Preview</h3>
            <div className="image-wrapper">
              {session.editPreview ? (
                <>
                  <img 
                    src={`data:image/png;base64,${session.editPreview}`}
                    alt="Preview"
                    className="display-image"
                  />
                  <div className="edit-controls">
                    <button onClick={acceptEdit} className="accept-btn">
                      Accept ✓
                    </button>
                    <button onClick={rejectEdit} className="reject-btn">
                      Reject ✗
                    </button>
                  </div>
                </>
              ) : (
                <div className="placeholder">Edit preview will appear here</div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;