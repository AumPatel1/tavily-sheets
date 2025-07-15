import React, { useState } from "react";
import { motion } from "framer-motion";
import { X, Loader2 } from "lucide-react";
import { SpreadsheetData } from "../types";

interface GenerateListModalProps {
  isOpen: boolean;
  onClose: () => void;
  onGenerate: (data: SpreadsheetData) => void;
  apiKey: string;
  checkApiKey: () => boolean | 0 | undefined;
  setToast: (toast: { message?: string; type?: "success" | "error" | "info"; isShowing?: boolean }) => void;
}

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

const GenerateListModal: React.FC<GenerateListModalProps> = ({
  isOpen,
  onClose,
  onGenerate,
  apiKey,
  checkApiKey,
  setToast,
}) => {
  const [prompt, setPrompt] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  const handleGenerate = async () => {
    if (!prompt.trim()) {
      setToast({
        message: "Please enter a prompt",
        type: "error",
        isShowing: true,
      });
      return;
    }

    if (!checkApiKey()) {
      setToast({
        message: "Please set a valid API Key",
        type: "error",
        isShowing: true,
      });
      return;
    }

    setIsLoading(true);

    try {
      const response = await fetch(`${API_URL}/api/generate-list`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: apiKey,
        },
        body: JSON.stringify({ prompt: prompt.trim() }),
      });

      if (!response.ok) {
        throw new Error("Failed to generate list");
      }

      const result = await response.json();
      
      if (!result.data) {
        throw new Error("Invalid response format");
      }

      // Handle new backend response structure
      if (result.data && result.data.headers && result.data.rows) {
        onGenerate({
          headers: result.data.headers,
          rows: result.data.rows,
        });
      } else if (Array.isArray(result.data)) {
        // fallback for old response
        const newData: SpreadsheetData = {
          headers: [
            { name: "Company/Entity" },
            { name: "Website" },
            { name: "Description" },
            { name: "Category" },
          ],
          rows: result.data.map((item: string) => [
            { value: item },
            { value: "" },
            { value: "" },
            { value: "" },
          ]),
        };
        onGenerate(newData);
      } else {
        setToast({
          message: "Invalid response from backend.",
          type: "error",
          isShowing: true,
        });
        onClose();
        return;
      }

      setToast({
        message: `Generated ${result.data.length} items successfully`,
        type: "success",
        isShowing: true,
      });
      onClose();
    } catch (error) {
      console.error("Error generating list:", error);
      setToast({
        message: "Failed to generate list. Please try again.",
        type: "error",
        isShowing: true,
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
      handleGenerate();
    }
  };

  if (!isOpen) return null;

  return (
    <motion.div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-50"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
    >
      <motion.div
        className="bg-white rounded-xl shadow-2xl w-full max-w-2xl mx-4 overflow-hidden"
        initial={{ scale: 0.9, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        exit={{ scale: 0.9, opacity: 0 }}
        transition={{ type: "spring", damping: 25, stiffness: 300 }}
      >
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-gray-200">
          <div>
            <h2 className="text-xl font-semibold text-gray-900">
              Generate List with AI
            </h2>
            <p className="text-sm text-gray-600 mt-1">
              Describe what kind of list you want to generate
            </p>
          </div>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-gray-600 transition-colors p-2 rounded-full hover:bg-gray-100"
          >
            <X size={20} />
          </button>
        </div>

        {/* Content */}
        <div className="p-6">
          <div className="mb-4">
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Prompt
            </label>
            <textarea
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="e.g., Who are the main competitors to Figma? List the top 10 SaaS companies in the project management space. What are the leading AI startups in 2024?"
              className="w-full h-32 p-3 border border-gray-300 rounded-lg resize-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              disabled={isLoading}
            />
            <p className="text-xs text-gray-500 mt-1">
              Press Cmd/Ctrl + Enter to generate
            </p>
          </div>

          <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 mb-6">
            <h3 className="text-sm font-medium text-blue-900 mb-2">
              What happens next?
            </h3>
            <ul className="text-sm text-blue-800 space-y-1">
              <li>• AI will generate a list based on your prompt</li>
              <li>• The list will be populated in the first column</li>
              <li>• You can then use "Enrich" features to add more data</li>
            </ul>
          </div>
        </div>

        {/* Footer */}
        <div className="flex items-center justify-end gap-3 p-6 border-t border-gray-200 bg-gray-50">
          <button
            onClick={onClose}
            className="px-4 py-2 text-gray-700 bg-white border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors"
            disabled={isLoading}
          >
            Cancel
          </button>
          <button
            onClick={handleGenerate}
            disabled={isLoading || !prompt.trim()}
            className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors flex items-center gap-2"
          >
            {isLoading ? (
              <>
                <Loader2 size={16} className="animate-spin" />
                Generating...
              </>
            ) : (
              "Generate List"
            )}
          </button>
        </div>
      </motion.div>
    </motion.div>
  );
};

export default GenerateListModal; 