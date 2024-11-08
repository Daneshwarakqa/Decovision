"use client";

import { useEffect, useState } from "react";

const History = () => {
  const [outputImage, setOutputImage] = useState<string | null>(null);
  // Retrieve the outputImage from session storage on component mount
  useEffect(() => {
    const savedImage = sessionStorage.getItem("outputImage");
    if (savedImage) {
      setOutputImage(savedImage);
    }
  }, []);
  return (
    <main className="flex min-h-screen flex-col py-10 lg:pl-72">
      <section className="mt-10 grid flex-1 gap-6 px-4 lg:px-6 xl:grid-cols-2 xl:gap-8 xl:px-8">
        {outputImage ? (
          <img
            src={outputImage}
            alt="Generated design"
            className="w-full max-w-md rounded-lg shadow-lg"
          />
        ) : (
          <p className="text-gray-500">
            No image generated yet. Please go back and generate a design.
          </p>
        )}
      </section>
    </main>
  );
};

export default History;
